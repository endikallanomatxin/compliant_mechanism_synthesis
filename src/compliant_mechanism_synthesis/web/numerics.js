/* These plots update with snapshots, independently of camera animation. */
let resolutionReport = null;
function numericalChart(id, xs, series, xLabel) {
    const svg = $(id);
    if (!xs.length) { svg.innerHTML = ''; return; }
    const values = series.flatMap(s => s.values);
    const low = Math.min(0, ...values), high = Math.max(...values, 1e-12);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const xy = (x, y) => [55 + (x - xmin) / Math.max(xmax - xmin, 1e-12) * 560, 165 - (y - low) / Math.max(high - low, 1e-12) * 125];
    const path = s => s.values.map((v, i) => xy(xs[i], v).join(',')).join(' ');
    svg.innerHTML = `<line x1="55" y1="165" x2="615" y2="165" stroke="#53697c"/><text x="8" y="42" fill="#9babba" font-size="11">${number(high)}</text><text x="8" y="165" fill="#9babba" font-size="11">${number(low)}</text>` + series.map((s, i) => `<text x="${55+i*260}" y="20" fill="${s.color}" font-size="12">${s.label}</text><polyline points="${path(s)}" fill="none" stroke="${s.color}" stroke-width="2"/>${s.values.map((v,j)=>{const p=xy(xs[j],v);return `<circle cx="${p[0]}" cy="${p[1]}" r="2" fill="${s.color}"/>`;}).join('')}`).join('') + `<text x="55" y="188" fill="#9babba" font-size="11">${number(xmin)}</text><text x="580" y="188" fill="#9babba" font-size="11">${number(xmax)}</text><text x="260" y="212" fill="#9babba" font-size="12">${xLabel}</text>`;
}
window.syncNumerics = function(state) {
    resolutionReport = null;
    $('resolutionPlot').setAttribute('hidden', '');
    $('resolutionErrorPlot').setAttribute('hidden', '');
    $('resolutionSummary').textContent = 'Comprueba la estructura seleccionada sin cambiar la búsqueda.';
    const history = state.gradient_history || [], last = history.at(-1);
    numericalChart('learningPlot', history.map(h=>h.iteration), [
        {label:'Tasa solicitada', color:'#f5cf79', values:history.map(h=>h.requested_rate)},
        {label:'Tasa media aceptada', color:'#6fe0ba', values:history.map(h=>h.mean_accepted_rate)}
    ], 'Iteración de búsqueda');
    numericalChart('improvementPlot', history.map(h=>h.iteration), [
        {label:'Mejora media del coste (%)', color:'#74b9ef', values:history.map(h=>h.improvement_percent)}
    ], 'Iteración de búsqueda');
    $('learningSummary').textContent = last ? `Último gradiente · iteración ${last.iteration}: ${last.accepted[0]}/${last.population} aceptan el paso inicial; ${last.unchanged} agotan los intentos. Mejora media del coste: ${number(last.improvement_percent)} % (${number(last.objective_before)} → ${number(last.objective_after)}). Tasa media entre pasos aceptados: ${number(last.mean_accepted_rate)}.` : 'Da un paso de gradiente para ver su aceptación y mejora.';
    $('learningTrials').innerHTML = last ? `<table><thead><tr><th>Tasa</th><th>Probadas</th><th>Aceptadas</th><th>Mejoran</th></tr></thead><tbody>${last.trials.map((count,i)=>`<tr><td>${number(last.requested_rate*.5**i)}</td><td>${count}</td><td>${last.accepted[i]}</td><td>${last.improved[i]}</td></tr>`).join('')}</tbody></table>` : '';
    numericalChart('newtonPlot', state.states.map(s=>s.factor), [{label:'Iteraciones de Newton',color:'#74b9ef',values:state.states.map(s=>s.iterations)}], 'Progreso del recorrido');
    numericalChart('stiffnessPlot', state.states.map(s=>s.factor), [
        {label:'Modo blando (N/m)',color:'#6fe0ba',values:state.states.map(s=>s.stiffness[0])},
        {label:'Siguiente modo (N/m)',color:'#f5cf79',values:state.states.map(s=>s.stiffness[1])}
    ], 'Progreso del recorrido natural');
    $('stiffnessSummary').textContent = `Longitud característica: ${number(state.continuation_config.characteristic_length*1000)} mm. Mayor relación rigidez blanda/transversal: ${number(Math.max(0,...state.states.map(s=>s.stiffness_ratio)))}; cerca de 0 indica un movimiento permitido bien separado, cerca de 1 indica varios movimientos blandos. El recorrido avanza por el modo menos rígido; la consigna solo elige su sentido y la longitud recorrida.`;
    const ratios = state.states.map(s=>s.residual/(state.continuation_config.tolerance*Math.max(1,Math.hypot(...s.reaction))));
    $('newtonSummary').textContent = `${state.continuation_config.steps} incrementos configurados · ${state.states.length} estados obtenidos · máximo ${Math.max(0,...state.states.map(s=>s.iterations))}/${state.continuation_config.max_iterations} iteraciones Newton · residuo/tolerancia máximo ${number(Math.max(0,...ratios))}. ${state.converged?'Recorrido completo.':'Recorrido incompleto.'} Se detiene ante un modo ambiguo, inestabilidad o fallo de equilibrio.`;
};
function drawResolution() {
    if (!resolutionReport?.converged) return;
    const r = resolutionReport, index = Number($('coordinate').value), dim = data.dimension;
    const multiplier = index < dim ? 1000 : 180/Math.PI;
    const names = data.dimension===2?['Ux','Uy','Theta']:['Ux','Uy','Uz','Rx','Ry','Rz'];
    $('resolutionPlot').removeAttribute('hidden');
    numericalChart('resolutionPlot', r.samples, [
        {label:`Actual · ${names[index]} (${index<dim?'mm':'°'})`,color:'#f5cf79',values:r.coarse.map(p=>p[index]*multiplier)},
        {label:'Doble de incrementos',color:'#6fe0ba',values:r.fine.map(p=>p[index]*multiplier)}
    ], 'Progreso del recorrido');
    $('resolutionErrorPlot').removeAttribute('hidden');
    numericalChart('resolutionErrorPlot', r.samples, [
        {label:`Diferencia · ${names[index]} (${index<dim?'mm':'°'})`,color:'#74b9ef',values:r.fine.map((p,i)=>(p[index]-r.coarse[i][index])*multiplier)}
    ], 'Progreso del recorrido');
}
$('coordinate').addEventListener('input', drawResolution);
$('checkResolution').onclick = async () => {
    if (busy || !data) return;
    running = false;
    $('run').textContent = 'Iniciar búsqueda';
    busy = true;
    document.querySelectorAll('button').forEach(b=>b.disabled=true);
    $('resolutionSummary').textContent = 'Calculando el recorrido con el doble de incrementos…';
    $('resolutionPlot').setAttribute('hidden', '');
    $('resolutionErrorPlot').setAttribute('hidden', '');
    try {
        resolutionReport = await request('/api/path-resolution', {index:data.selected_index});
        const r = resolutionReport;
        $('resolutionPlot').setAttribute('hidden', '');
        $('resolutionSummary').textContent = r.converged ? `${r.coarse_steps} → ${r.fine_steps} incrementos (${r.coarse_states} → ${r.fine_states} estados). Diferencia máxima: ${number(r.position_max_mm)} mm y ${number(r.orientation_max_deg)}°. Diferencia RMS normalizada: ${number(r.weighted_rms_difference)}. Coste integrado de seguimiento: ${number(r.coarse_tracking_cost)} → ${number(r.fine_tracking_cost)} (${number(r.cost_change_percent)} %). ${number(r.seconds)} s.` : `El recorrido refinado no converge (${r.fine_states} estados). La resolución actual no queda validada.`;
        drawResolution();
    } catch(e) { $('resolutionSummary').textContent=e.message; }
    finally { busy=false; document.querySelectorAll('button').forEach(b=>b.disabled=false); }
};
if (data) window.syncNumerics(data);
