(function () {
  'use strict';

  const STORAGE_KEY_TASKS = 'task_board_tasks';
  const STORAGE_KEY_FILTER = 'task_board_filter';

  let tasks = [];
  let currentFilter = 'all';

  // DOM elements
  const taskForm = document.getElementById('task-form');
  const taskInput = document.getElementById('task-input');
  const taskList = document.getElementById('task-list');
  const filterBtns = document.querySelectorAll('.filter-btn');
  const clearCompletedBtn = document.getElementById('clear-completed-btn');
  const activeCountEl = document.getElementById('active-count');
  const totalCountEl = document.getElementById('total-count');
  const statusLiveEl = document.getElementById('status-live');

  function announce(message) {
    if (statusLiveEl) {
      statusLiveEl.textContent = '';
      setTimeout(() => {
        statusLiveEl.textContent = message;
      }, 50);
    }
  }

  function loadState() {
    try {
      const savedTasks = localStorage.getItem(STORAGE_KEY_TASKS);
      tasks = savedTasks ? JSON.parse(savedTasks) : [];
      if (!Array.isArray(tasks)) {
        tasks = [];
      }
    } catch (e) {
      tasks = [];
    }

    try {
      const savedFilter = localStorage.getItem(STORAGE_KEY_FILTER);
      if (savedFilter && ['all', 'active', 'completed'].includes(savedFilter)) {
        currentFilter = savedFilter;
      } else {
        currentFilter = 'all';
      }
    } catch (e) {
      currentFilter = 'all';
    }
  }

  function saveTasks() {
    try {
      localStorage.setItem(STORAGE_KEY_TASKS, JSON.stringify(tasks));
    } catch (e) {
      console.error('Failed to save tasks to localStorage', e);
    }
  }

  function saveFilter() {
    try {
      localStorage.setItem(STORAGE_KEY_FILTER, currentFilter);
    } catch (e) {
      console.error('Failed to save filter to localStorage', e);
    }
  }

  function updateCounts() {
    const totalCount = tasks.length;
    const activeCount = tasks.filter(t => !t.completed).length;
    const completedCount = totalCount - activeCount;

    activeCountEl.textContent = String(activeCount);
    totalCountEl.textContent = String(totalCount);

    clearCompletedBtn.disabled = completedCount === 0;
  }

  function updateFilterButtons() {
    filterBtns.forEach(btn => {
      const filterValue = btn.getAttribute('data-filter');
      const isSelected = filterValue === currentFilter;
      btn.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
      if (isSelected) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  function renderTasks() {
    taskList.innerHTML = '';

    const filteredTasks = tasks.filter(task => {
      if (currentFilter === 'active') return !task.completed;
      if (currentFilter === 'completed') return task.completed;
      return true;
    });

    filteredTasks.forEach(task => {
      const li = document.createElement('li');
      li.className = 'task-item' + (task.completed ? ' completed' : '');
      li.setAttribute('data-id', task.id);

      const contentDiv = document.createElement('div');
      contentDiv.className = 'task-item-content';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'task-checkbox';
      checkbox.checked = Boolean(task.completed);
      checkbox.id = 'task-check-' + task.id;
      checkbox.setAttribute('aria-label', `Mark "${task.text}" as ${task.completed ? 'active' : 'completed'}`);

      const label = document.createElement('label');
      label.htmlFor = 'task-check-' + task.id;
      label.className = 'task-text';
      label.textContent = task.text;

      contentDiv.appendChild(checkbox);
      contentDiv.appendChild(label);

      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'delete-btn';
      deleteBtn.setAttribute('aria-label', `Delete task "${task.text}"`);
      deleteBtn.textContent = 'Delete';

      checkbox.addEventListener('change', () => {
        toggleTask(task.id);
      });

      deleteBtn.addEventListener('click', () => {
        deleteTask(task.id);
      });

      li.appendChild(contentDiv);
      li.appendChild(deleteBtn);
      taskList.appendChild(li);
    });

    updateCounts();
    updateFilterButtons();
  }

  function addTask(text) {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    const newTask = { 
      id: Date.now().toString(36) + Math.random().toString(36).substring(2, 7),
      text: trimmed,
      completed: false
    };

    tasks.push(newTask);
    saveTasks();
    renderTasks();
    announce(`Added task: ${newTask.text}`);
  }

  function toggleTask(id) {
    const task = tasks.find(t => t.id === id);
    if (!task) return;

    task.completed = !task.completed;
    saveTasks();
    renderTasks();
    announce(`Marked "${task.text}" as ${task.completed ? 'completed' : 'active'}`);
  }

  function deleteTask(id) {
    const index = tasks.findIndex(t => t.id === id);
    if (index === -1) return;

    const [deleted] = tasks.splice(index, 1);
    saveTasks();
    renderTasks();
    announce(`Deleted task: ${deleted.text}`);
  }

  function clearCompleted() {
    const completedCount = tasks.filter(t => t.completed).length;
    if (completedCount === 0) return;

    tasks = tasks.filter(t => !t.completed);
    saveTasks();
    renderTasks();
    announce(`Cleared ${completedCount} completed task${completedCount === 1 ? '' : 's'}`);
  }

  function setFilter(filter) {
    if (!['all', 'active', 'completed'].includes(filter)) return;
    currentFilter = filter;
    saveFilter();
    renderTasks();
    announce(`Showing ${filter} tasks`);
  }

  // Event Listeners
  taskForm.addEventListener('submit', function (e) {
    e.preventDefault();
    const value = taskInput.value;
    if (!value || !value.trim()) {
      taskInput.focus();
      return;
    }
    addTask(value);
    taskInput.value = '';
    taskInput.focus();
  });

  filterBtns.forEach(btn => {
    btn.addEventListener('click', function () {
      const filter = this.getAttribute('data-filter');
      setFilter(filter);
    });
  });

  clearCompletedBtn.addEventListener('click', function () {
    clearCompleted();
  });

  // Initialization
  loadState();
  renderTasks();
})();
