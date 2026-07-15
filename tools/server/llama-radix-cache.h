#pragma once

#include "common.h"
#include <vector>
#include <memory>
#include <map>
#include <chrono>
#include <limits>

struct radix_checkpoint_node {
    std::vector<llama_token> tokens; // The suffix tokens represented by this node
    std::shared_ptr<common_prompt_checkpoint> checkpoint; // Optional checkpoint data (only present on completed nodes)
    
    std::map<llama_token, std::shared_ptr<radix_checkpoint_node>> children;
    
    int64_t last_accessed_us = 0; // Timestamp for LRU eviction
    
    radix_checkpoint_node() {
        last_accessed_us = ggml_time_us();
    }
    
    size_t size_bytes() const {
        size_t s = tokens.size() * sizeof(llama_token);
        if (checkpoint) {
            s += checkpoint->size();
        }
        return s;
    }
};

class radix_checkpoint_tree {
public:
    std::shared_ptr<radix_checkpoint_node> root;
    size_t max_size_bytes = 0;
    
    radix_checkpoint_tree(size_t max_size_bytes) : max_size_bytes(max_size_bytes) {
        root = std::make_shared<radix_checkpoint_node>();
    }
    
    // Find the longest prefix in the tree that matches the input tokens.
    // Returns the matching node, and the number of tokens matched.
    std::shared_ptr<radix_checkpoint_node> find_longest_prefix(
        const std::vector<llama_token>& query_tokens, 
        size_t& matched_len) {
        
        std::shared_ptr<radix_checkpoint_node> curr = root;
        matched_len = 0;
        std::shared_ptr<radix_checkpoint_node> best_match = nullptr;
        size_t best_match_len = 0;
        
        size_t i = 0;
        while (i < query_tokens.size()) {
            llama_token next_tok = query_tokens[i];
            auto it = curr->children.find(next_tok);
            if (it == curr->children.end()) {
                break;
            }
            
            curr = it->second;
            // Match suffix of curr node
            size_t node_toks_len = curr->tokens.size();
            bool match = true;
            for (size_t j = 0; j < node_toks_len; ++j) {
                if (i + j >= query_tokens.size() || query_tokens[i + j] != curr->tokens[j]) {
                    match = false;
                    break;
                }
            }
            
            if (!match) {
                break;
            }
            
            i += node_toks_len;
            curr->last_accessed_us = ggml_time_us();
            if (curr->checkpoint) {
                best_match = curr;
                best_match_len = i;
            }
        }
        
        matched_len = best_match_len;
        return best_match;
    }
    
    // Insert a checkpoint into the radix tree for a given sequence of tokens.
    void insert(const std::vector<llama_token>& path_tokens, const common_prompt_checkpoint& ckpt) {
        if (path_tokens.empty()) return;
        
        std::shared_ptr<radix_checkpoint_node> curr = root;
        size_t i = 0;
        
        while (i < path_tokens.size()) {
            llama_token next_tok = path_tokens[i];
            auto it = curr->children.find(next_tok);
            
            if (it == curr->children.end()) {
                // Insert a new leaf node containing the rest of the path
                auto new_node = std::make_shared<radix_checkpoint_node>();
                new_node->tokens = std::vector<llama_token>(path_tokens.begin() + i, path_tokens.end());
                curr->children[next_tok] = new_node;
                curr = new_node;
                break;
            }
            
            std::shared_ptr<radix_checkpoint_node> child = it->second;
            // Find common prefix between child->tokens and path_tokens[i...]
            size_t common_len = 0;
            size_t child_len = child->tokens.size();
            size_t path_len = path_tokens.size() - i;
            
            while (common_len < child_len && common_len < path_len && 
                   child->tokens[common_len] == path_tokens[i + common_len]) {
                common_len++;
            }
            
            if (common_len == child_len) {
                // Go down to child node
                curr = child;
                i += child_len;
            } else {
                // Split the child node at common_len
                auto split_node = std::make_shared<radix_checkpoint_node>();
                split_node->tokens = std::vector<llama_token>(child->tokens.begin(), child->tokens.begin() + common_len);
                
                // Adjust original child tokens to be the suffix after split
                child->tokens = std::vector<llama_token>(child->tokens.begin() + common_len, child->tokens.end());
                
                // Move child under split_node
                split_node->children[child->tokens[0]] = child;
                
                // Replace child under curr with split_node
                curr->children[next_tok] = split_node;
                
                curr = split_node;
                i += common_len;
                
                // Insert the remaining path_tokens under split_node
                if (i < path_tokens.size()) {
                    auto new_node = std::make_shared<radix_checkpoint_node>();
                    new_node->tokens = std::vector<llama_token>(path_tokens.begin() + i, path_tokens.end());
                    curr->children[path_tokens[i]] = new_node;
                    curr = new_node;
                }
                break;
            }
        }
        
        curr->checkpoint = std::make_shared<common_prompt_checkpoint>(ckpt);
        curr->last_accessed_us = ggml_time_us();
        
        evict_if_needed();
    }
    
    size_t calculate_total_size(std::shared_ptr<radix_checkpoint_node> node) const {
        size_t s = node->size_bytes();
        for (const auto& pair : node->children) {
            s += calculate_total_size(pair.second);
        }
        return s;
    }
    
    void evict_if_needed() {
        if (max_size_bytes == 0) return;
        
        while (calculate_total_size(root) > max_size_bytes) {
            std::shared_ptr<radix_checkpoint_node> oldest_node = nullptr;
            int64_t oldest_time = std::numeric_limits<int64_t>::max();
            std::shared_ptr<radix_checkpoint_node> oldest_parent = nullptr;
            llama_token oldest_key = 0;
            
            find_oldest_leaf(root, nullptr, 0, oldest_node, oldest_time, oldest_parent, oldest_key);
            
            if (oldest_node && oldest_parent) {
                // Erase the leaf node
                oldest_parent->children.erase(oldest_key);
            } else if (oldest_node) {
                // Root is the only node and it has a checkpoint
                oldest_node->checkpoint.reset();
                break;
            } else {
                break;
            }
        }
    }
    
private:
    void find_oldest_leaf(std::shared_ptr<radix_checkpoint_node> curr, 
                          std::shared_ptr<radix_checkpoint_node> parent, 
                          llama_token key_in_parent,
                          std::shared_ptr<radix_checkpoint_node>& oldest_node,
                          int64_t& oldest_time,
                          std::shared_ptr<radix_checkpoint_node>& oldest_parent,
                          llama_token& oldest_key) {
        
        if (curr->children.empty()) {
            if (curr->checkpoint && curr->last_accessed_us < oldest_time) {
                oldest_time = curr->last_accessed_us;
                oldest_node = curr;
                oldest_parent = parent;
                oldest_key = key_in_parent;
            }
            return;
        }
        
        for (const auto& pair : curr->children) {
            find_oldest_leaf(pair.second, curr, pair.first, oldest_node, oldest_time, oldest_parent, oldest_key);
        }
    }
};
