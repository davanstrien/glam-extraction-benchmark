const body = document.getElementById('results');
const buttons = [...document.querySelectorAll('[data-column]')];
const sizeFilter = document.getElementById('max-params');
let activeColumn = 1;
let direction = 'asc';

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

sizeFilter.onchange = () => {
  let visible = 0;
  [...body.rows].forEach(row => {
    row.hidden = sizeFilter.value !== 'all'
      && (row.dataset.params === '' || Number(row.dataset.params) > Number(sizeFilter.value));
    if (!row.hidden) visible++;
  });
  document.getElementById('model-count').textContent = `${visible} of ${body.rows.length} models`;
  document.getElementById('empty-results').hidden = visible !== 0;
};
sortRows();
sizeFilter.onchange();
