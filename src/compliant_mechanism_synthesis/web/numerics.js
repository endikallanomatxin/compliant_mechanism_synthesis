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
    svg.innerHTML = `<line x1="55" y1="165" x2="615" y2="165" stroke="#cfd6d0"/><text x="8" y="42" fill="#6b756f" font-size="11">${number(high)}</text><text x="8" y="165" fill="#6b756f" font-size="11">${number(low)}</text>` + series.map((s, i) => `<text x="${55+i*(series.length>2?185:260)}" y="20" fill="${s.color}" font-size="12">${s.label}</text><polyline points="${path(s)}" fill="none" stroke="${s.color}" stroke-width="2"/>${s.values.map((v,j)=>{const p=xy(xs[j],v);return `<circle cx="${p[0]}" cy="${p[1]}" r="2" fill="${s.color}"/>`;}).join('')}`).join('') + `<text x="55" y="188" fill="#6b756f" font-size="11">${number(xmin)}</text><text x="580" y="188" fill="#6b756f" font-size="11">${number(xmax)}</text><text x="260" y="212" fill="#6b756f" font-size="12">${xLabel}</text>`;
}
window.syncNumerics = function(state) {
    resolutionReport = null;
    $('resolutionPlot').setAttribute('hidden', '');
    $('resolutionErrorPlot').setAttribute('hidden', '');
    $('resolutionSummary').textContent = 'Comprueba la estructura seleccionada sin cambiar la búsqueda.';
    const history = state.gradient_history || [], last = history.at(-1);
    numericalChart('learningPlot', history.map(h=>h.sequence), [
        {label:'Tasa solicitada', color:'#aa7022', values:history.map(h=>h.requested_rate)},
        {label:'Tasa media aceptada', color:'#23786c', values:history.map(h=>h.mean_accepted_rate)},
        {label:'Siguiente paso', color:'#456f9b', values:history.map(h=>h.next_rate)}
    ], 'Pasos de gradiente ejecutados');
    numericalChart('monotonicityPlot', history.map(h=>h.sequence), [
        {label:'Mejoran en el primer intento (%)',color:'#23786c',values:history.map(h=>100*h.monotonic_fraction)},
        {label:'Objetivo (%)',color:'#aa7022',values:history.map(h=>100*h.monotonic_target)}
    ], 'Pasos de gradiente ejecutados');
    numericalChart('improvementPlot', history.map(h=>h.sequence), [
        {label:'Mejora media del coste (%)', color:'#456f9b', values:history.map(h=>h.improvement_percent)}
    ], 'Pasos de gradiente ejecutados');
    $('learningSummary').textContent = last ? `Último gradiente · ${last.phase==='offspring'?'descendientes antes de selección':'población'} · iteración ${last.iteration}: ${last.initial_monotonic}/${last.eligible} mejoran en el primer intento (${(100*last.monotonic_fraction).toFixed(1)} %, objetivo ${(100*last.monotonic_target).toFixed(0)} %); ${last.accepted[0]}/${last.eligible} aceptan el paso inicial; ${last.unchanged} agotan los intentos. Mejora media del coste: ${number(last.improvement_percent)} % (${number(last.objective_before)} → ${number(last.objective_after)}). Tasa usada: ${number(last.requested_rate)} → siguiente: ${number(last.next_rate)} (${last.adaptive_rate?(last.next_rate>last.requested_rate?'sube':last.next_rate<last.requested_rate?'baja':'se mantiene'):'manual'}).` : 'Da un paso de gradiente para ver su aceptación y mejora.';
    $('learningRateStatus').textContent=`Próxima tasa: ${number(state.current_learning_rate)} · ${state.search_config.adaptive_learning_rate?'adaptativa':'manual'}`;
    $('learningTrials').innerHTML = last ? `<table><thead><tr><th>Tasa</th><th>Probadas</th><th>Aceptadas</th><th>Mejoran</th></tr></thead><tbody>${last.trials.map((count,i)=>`<tr><td>${number(last.requested_rate*.5**i)}</td><td>${count}</td><td>${last.accepted[i]}</td><td>${last.improved[i]}</td></tr>`).join('')}</tbody></table>` : '';
    numericalChart('newtonPlot', state.states.map(s=>s.factor), [{label:'Iteraciones de Newton',color:'#456f9b',values:state.states.map(s=>s.iterations)}], 'Progreso del recorrido');
    numericalChart('stiffnessPlot', state.states.map(s=>s.factor), [
        {label:'Pasivo · mínima (N/m)',color:'#23786c',values:state.states.map(s=>s.stiffness[0])},
        {label:'Pasivo · siguiente (N/m)',color:'#aa7022',values:state.states.map(s=>s.stiffness[1])}
    ], 'Progreso de entrada');
    $('stiffnessSummary').textContent = `Rigidez del pasivo bloqueando el activo y relajando los nodos libres. Escala para comparar giro y traslación: ${number(state.continuation_config.characteristic_length*1000)} mm. El activo sigue la entrada impuesta; el pasivo resulta del equilibrio no lineal.`;
    const ratios = state.states.map(s=>s.residual/(state.continuation_config.tolerance*Math.max(1,Math.hypot(...s.reaction))));
    $('newtonSummary').textContent = `${state.continuation_config.steps} incrementos configurados · ${state.states.length} estados obtenidos · máximo ${Math.max(0,...state.states.map(s=>s.iterations))}/${state.continuation_config.max_iterations} iteraciones Newton · residuo/tolerancia máximo ${number(Math.max(0,...ratios))}. ${state.converged?'Recorrido completo.':'Recorrido incompleto.'} Se subdivide la entrada al fallar Newton; se detiene ante pérdida de estabilidad o falta de convergencia.`;
};
function drawResolution() {
    if (!resolutionReport?.converged) return;
    const r = resolutionReport;
    const names = ['Ux','Uy','Uz','Rx','Ry','Rz'];
    $('resolutionPlot').removeAttribute('hidden');
    $('resolutionErrorPlot').removeAttribute('hidden');
    names.forEach((name,index) => {
        const multiplier = index < 3 ? 1000 : 180/Math.PI;
        numericalChart('resolutionCoordinate'+index, r.samples, [
            {label:'Actual',color:'#aa7022',values:r.coarse.map(p=>p[index]*multiplier)},
            {label:r.coarse_model==='linear'?'No lineal':'Recorrido refinado',color:'#23786c',values:r.fine.map(p=>p[index]*multiplier)}
        ], 'Progreso del recorrido');
        numericalChart('resolutionDifference'+index, r.samples, [
            {label:`Diferencia · ${name} (${index<3?'mm':'°'})`,color:'#456f9b',values:r.fine.map((p,i)=>(p[index]-r.coarse[i][index])*multiplier)}
        ], 'Progreso del recorrido');
    });
}
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
