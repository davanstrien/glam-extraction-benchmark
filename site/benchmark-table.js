const body = document.getElementById('results');
const buttons = [...document.querySelectorAll('[data-column]')];
const sizeButtons = [...document.querySelectorAll('[data-max-params]')];
// Configs without identifier fields start with extraction F1.
const f1Column = Number(body.dataset.f1Column || 4);
let activeColumn = Number(body.dataset.defaultColumn ||
  ([...body.rows].some(row => row.cells[1].dataset.value !== '') ? 1 : f1Column));
let direction = activeColumn === f1Column ? 'desc' : 'asc';

function sortRows() {
  const sign = direction === 'asc' ? 1 : -1;
  [...body.rows].sort((a, b) => {
    const x = a.cells[activeColumn].dataset.value;
    const y = b.cells[activeColumn].dataset.value;
    // Missing rates stay last in either direction.
    if (x === '' || y === '') return (x === '') - (y === '');
    const comparison = activeColumn === 0
      ? x.localeCompare(y, 'en', { numeric: true, sensitivity: 'base' })
      : Number(x) - Number(y);
    return sign * comparison || a.cells[0].dataset.value.localeCompare(b.cells[0].dataset.value);
  }).forEach(row => body.append(row));
  buttons.forEach(button => {
    const active = Number(button.dataset.column) === activeColumn;
    button.parentElement.removeAttribute('aria-sort');
    if (active) button.parentElement.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');
    button.querySelector('span').textContent = active ? (direction === 'asc' ? '↑' : '↓') : '↕';
  });
}

buttons.forEach(button => button.onclick = () => {
  const column = Number(button.dataset.column);
  direction = column === activeColumn ? (direction === 'asc' ? 'desc' : 'asc') : button.dataset.direction;
  activeColumn = column;
  sortRows();
});

function filterSize(limit) {
  let visible = 0;
  [...body.rows].forEach(row => {
    row.hidden = limit !== 'all'
      && (row.dataset.params === '' || Number(row.dataset.params) >= Number(limit));
    if (!row.hidden) visible++;
  });
  document.getElementById('model-count').textContent = `${visible} of ${body.rows.length} models`;
  document.getElementById('empty-results').hidden = visible !== 0;
  sizeButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.maxParams === limit)));
  drawSizePlot([...body.rows]);
}
// Keep shareable state in the URL, including when embedded on the Hub.
const datasetSelect = document.getElementById('dataset');
function syncUrl(limit) {
  const url = new URL(window.location.href);
  if (limit === 'all') url.searchParams.delete('max_params');
  else url.searchParams.set('max_params', limit);
  url.searchParams.set('dataset', datasetSelect.selectedOptions[0].dataset.config);
  window.history.replaceState(null, '', url);
  window.parent.postMessage({queryString: url.search}, 'https://huggingface.co');
}
function navigateDataset(option, replace = false) {
  const url = new URL(option.value, window.location.href);
  url.search = window.location.search;
  url.searchParams.set('dataset', option.dataset.config);
  url.hash = window.location.hash;
  if (replace) window.location.replace(url);
  else window.location.assign(url);
}
datasetSelect.addEventListener('change', () => navigateDataset(datasetSelect.selectedOptions[0]));
sizeButtons.forEach(button => button.onclick = () => {
  filterSize(button.dataset.maxParams);
  syncUrl(button.dataset.maxParams);
});
function restoreUrl() {
  const params = new URLSearchParams(window.location.search);
  const requested = [...datasetSelect.options].find(option => option.dataset.config === params.get('dataset'));
  if (requested && !requested.selected) {
    navigateDataset(requested, true);
    return;
  }
  const value = params.get('max_params');
  const limit = sizeButtons.some(button => button.dataset.maxParams === value) ? value : 'all';
  filterSize(limit);
  syncUrl(limit);
}
window.addEventListener('popstate', restoreUrl);
sortRows();
restoreUrl();
