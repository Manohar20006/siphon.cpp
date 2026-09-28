const STORAGE_KEY = 'focus-board.tasks.v1';
const FILTER_KEY = 'focus-board.filter.v1';
const form = document.querySelector('#task-form');
const input = document.querySelector('#task-input');
const error = document.querySelector('#form-error');
const list = document.querySelector('#task-list');
const empty = document.querySelector('#empty-state');
const activeCount = document.querySelector('#active-count');
const totalCount = document.querySelector('#total-count');
const clearButton = document.querySelector('#clear-completed');
const status = document.querySelector('#status');
const filterButtons = [...document.querySelectorAll('[data-filter]')];

let tasks = readTasks();
let filter = ['all', 'active', 'completed'].includes(localStorage.getItem(FILTER_KEY)) ? localStorage.getItem(FILTER_KEY) : 'all';

function readTasks() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(value) ? value.filter(task => task && typeof task.text === 'string') : [];
  } catch { return []; }
}

function save() { localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks)); }
function announce(message) { status.textContent = message; }

function visibleTasks() {
  if (filter === 'active') return tasks.filter(task => !task.completed);
  if (filter === 'completed') return tasks.filter(task => task.completed);
  return tasks;
}

function render() {
  list.replaceChildren();
  for (const task of visibleTasks()) {
    const item = document.createElement('li');
    item.className = `task${task.completed ? ' completed' : ''}`;
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = task.completed;
    checkbox.setAttribute('aria-label', `Mark ${task.text} ${task.completed ? 'active' : 'completed'}`);
    checkbox.addEventListener('change', () => toggleTask(task.id));
    const label = document.createElement('label');
    label.textContent = task.text;
    const remove = document.createElement('button');
    remove.type = 'button'; remove.className = 'delete'; remove.textContent = 'Delete';
    remove.setAttribute('aria-label', `Delete ${task.text}`);
    remove.addEventListener('click', () => deleteTask(task.id));
    item.append(checkbox, label, remove); list.append(item);
  }
  empty.hidden = visibleTasks().length !== 0;
  activeCount.textContent = tasks.filter(task => !task.completed).length;
  totalCount.textContent = tasks.length;
  clearButton.disabled = !tasks.some(task => task.completed);
  filterButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filter === filter)));
}

function toggleTask(id) { tasks = tasks.map(task => task.id === id ? {...task, completed: !task.completed} : task); save(); render(); announce('Task status updated.'); }
function deleteTask(id) { const task = tasks.find(item => item.id === id); tasks = tasks.filter(item => item.id !== id); save(); render(); announce(`${task?.text || 'Task'} deleted.`); }

form.addEventListener('submit', event => {
  event.preventDefault(); const text = input.value.trim();
  if (!text) { error.textContent = 'Enter a task before adding it.'; input.focus(); return; }
  error.textContent = ''; tasks.push({id: crypto.randomUUID(), text, completed: false}); save(); render(); input.value = ''; input.focus(); announce(`${text} added.`);
});

filterButtons.forEach(button => button.addEventListener('click', () => { filter = button.dataset.filter; localStorage.setItem(FILTER_KEY, filter); render(); announce(`Showing ${filter} tasks.`); }));
clearButton.addEventListener('click', () => { const count = tasks.filter(task => task.completed).length; tasks = tasks.filter(task => !task.completed); save(); render(); announce(`${count} completed task${count === 1 ? '' : 's'} cleared.`); });
render();
