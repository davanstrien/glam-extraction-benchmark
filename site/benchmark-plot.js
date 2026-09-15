function drawSizePlot(rows) {
  const svg = document.getElementById('size-plot');
  const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const points = rows.filter(row => !row.hidden && Number(row.dataset.params) > 0).map(row => ({
    name: row.cells[0].dataset.value,
    params: Number(row.dataset.params),
    f1: Number(row.cells[Number(svg.dataset.f1Column || 4)].dataset.value),
    href: row.cells[0].querySelector('a')?.getAttribute('href'),
  }));
  const omitted = rows.filter(row => !row.hidden).length - points.length;
  document.getElementById('plot-note').textContent =
    'Each point is a model; hover or focus for details, click to open its Hub page.'
    + (omitted ? ` ${omitted} model(s) with unknown size omitted.` : '');
  if (!points.length) {
    svg.innerHTML = '<text x="380" y="180" text-anchor="middle">No models with a known size match this filter.</text>';
    return;
  }
  // Keep the axes stable when the size filter changes.
  const sizes = rows.map(row => Number(row.dataset.params)).filter(size => size > 0);
  const ticks = [0.01,0.025,0.05,0.1,0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024];
  const lo = [...ticks].reverse().find(t => t <= Math.min(...sizes)) ?? Math.min(...sizes) / 2;
  const hi = ticks.find(t => t > Math.max(...sizes)) ?? Math.max(...sizes) * 2;
  const left = 58, top = 20, width = 678, height = 284;
  const X = params => left + (Math.log10(params) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * width;
  const Y = f1 => top + (1 - f1) * height;
  const frontier = points.filter(p => !points.some(q =>
    q.params <= p.params && q.f1 >= p.f1 && (q.params < p.params || q.f1 > p.f1)
  )).sort((a,b) => a.params - b.params);
  let chart = '';
  for (let value = 0; value <= 100; value += 20) {
    const y = Y(value / 100);
    chart += `<line class="plot-grid" x1="${left}" y1="${y}" x2="${left+width}" y2="${y}"/>`
      + `<text class="plot-tick" x="${left-10}" y="${y+4}" text-anchor="end">${value}</text>`;
  }
  ticks.filter(t => t >= lo && t <= hi).forEach(t => {
    const x = X(t);
    chart += `<line class="plot-grid" x1="${x}" y1="${top}" x2="${x}" y2="${top+height}"/>`
      + `<text class="plot-tick" x="${x}" y="${top+height+20}" text-anchor="middle">${t}B</text>`;
  });
  chart += `<text class="plot-axis" x="${left+width/2}" y="350" text-anchor="middle">Total parameters (billions · log scale)</text>`
    + '<text class="plot-axis" transform="translate(16,162) rotate(-90)" text-anchor="middle">Extraction F1 (0–100)</text>';
  if (frontier.length > 1) chart += `<polyline class="plot-frontier" points="${frontier.map(p => `${X(p.params)},${Y(p.f1)}`).join(' ')}"/>`;
  points.forEach(p => {
    const best = frontier.includes(p), x = X(p.params), y = Y(p.f1);
    const description = `${p.name}: ${p.params}B total parameters, F1 ${(p.f1*100).toFixed(1)}${best ? ', on the observed Pareto frontier' : ''}`;
    const opening = p.href ? `<a href="${escapeText(p.href)}" target="_blank" rel="noopener noreferrer" aria-label="${escapeText(description)}">`
      : `<g tabindex="0" role="img" aria-label="${escapeText(description)}">`;
    chart += `${opening}<title>${escapeText(description)}</title><circle class="plot-point${best ? ' frontier-point' : ''}" cx="${x}" cy="${y}" r="6"/>${p.href ? '</a>' : '</g>'}`;
    if (best) {
      const right = x < left + width * 0.65;
      chart += `<text class="plot-label" x="${x+(right ? 10 : -10)}" y="${y-11}" text-anchor="${right ? 'start' : 'end'}">${escapeText(p.name)}</text>`;
    }
  });
  svg.innerHTML = chart;
}
