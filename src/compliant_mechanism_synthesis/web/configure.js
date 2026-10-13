/* Specification editor keeps a local draft until validation succeeds. */
let draft = null;

function namesFor(dim) {
    return dim === 2 ? ['Ux', 'Uy', 'Theta'] : ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz'];
}

function screen(name) {
    const spec = name === 'spec';
    $('specScreen').hidden = !spec;
    $('synthesisScreen').hidden = spec;
    $('navSpec').classList.toggle('active', spec);
    $('navSynthesis').classList.toggle('active', !spec);
    if (spec) {
        running = false;
        playing = false;
        $('run').textContent = 'Iniciar búsqueda';
        $('play').textContent = 'Animar';
    } else requestAnimationFrame(render);
}
window.syncSpecification = function(state) {
    if (draft) return;
    draft = structuredClone(state.specification);
    draft.dimension = state.dimension;
    draft.factors = [...state.waypoints.factors];
    draft.poses = structuredClone(state.waypoints.poses);
    draft.rest_factor = state.waypoints.rest_factor;
    $("restFactor").value=draft.rest_factor;
    draft.radiusBounds = state.search_config;
    $('dimension').value = state.dimension;
    drawEditor();
    $('specDigest').textContent = $('specSummary').textContent;
};

function readRows() {
    if (!draft) return;
    document.querySelectorAll('#nodeRows tr').forEach((row, i) => {
        draft.positions[i] = [...row.querySelectorAll('input')].map(input => Number(input.value));
        draft.roles[i] = row.querySelector('select').value;
    });
    draft.edges = [];
    draft.radii = [];
    document.querySelectorAll('.beam-row').forEach(row => {
        if (row.querySelector('[type=checkbox]').checked) {
            draft.edges.push(row.dataset.edge.split(',').map(Number));
            draft.radii.push(Number(row.querySelector('[type=number]').value) / 1000);
        }
    });
    draft.factors = [];
    draft.poses = [];
    document.querySelectorAll('#pathRows tr').forEach(row => {
        const values = [...row.querySelectorAll('input')].map(input => Number(input.value));
        draft.factors.push(values[0]);
        draft.poses.push(values.slice(1).map((v, k) => k < draft.dimension ? v / 1000 : v * Math.PI / 180));
    });
    draft.rest_factor = Number($('restFactor').value);
    preview();
}

function drawEditor() {
    const dim = draft.dimension,
        axes = ['X', 'Y', 'Z'].slice(0, dim),
        names = namesFor(dim);
    $('nodeHead').innerHTML = '<tr><th>Nodo</th>' + axes.map(a => `<th>${a} (m)</th>`).join('') + '<th>Rol</th><th></th></tr>';
    $('nodeRows').innerHTML = draft.positions.map((p, i) => `<tr><td>${i}</td>${p.map(v=>`<td><input type="number" step="0.001" value="${v}" aria-label="Nodo ${i} coordenada"></td>`).join('')}<td><select aria-label="Rol nodo ${i}">${['fixed','mobile','free'].map((r,k)=>`<option value="${r}" ${draft.roles[i]===r?'selected':''}>${['Fijo','Móvil','Libre'][k]}</option>`).join('')}</select></td><td><button class="remove-node" data-index="${i}" aria-label="Eliminar nodo ${i}">×</button></td></tr>`).join('');
    const enabled = new Map(draft.edges.map((e, i) => [
        [...e].sort((a, b) => a - b).join(','), draft.radii[i] * 1000
    ]));
    let beams = '';
    for (let i = 0; i < draft.positions.length; i++)
        for (let j = i + 1; j < draft.positions.length; j++) {
            if (draft.roles[i] === draft.roles[j] && ['fixed', 'mobile'].includes(draft.roles[i])) continue;
            const key = [i, j].join(','),
                radius = enabled.get(key);
            beams += `<label class="beam-row" data-edge="${key}"><input type="checkbox" ${radius!==undefined?'checked':''}>${i}–${j}<input type="number" min="${draft.radiusBounds.radius_min*1000+.001}" max="${draft.radiusBounds.radius_max*1000-.001}" step="0.05" value="${radius??draft.radiusBounds.initial_radius*1000}" aria-label="Radio viga ${i} ${j}"></label>`;
        }
    $('beamRows').innerHTML = beams;
    $('pathHead').innerHTML = '<tr><th>Progreso 0–1</th>' + names.map((n, k) => `<th>${n} (${k<dim?'mm':'°'})</th>`).join('') + '<th></th></tr>';
    $('pathRows').innerHTML = draft.factors.map((t, i) => `<tr><td><input type="number" min="0" max="1" step="0.05" value="${t}" aria-label="Progreso punto ${i}"></td>${draft.poses[i].map((v,k)=>`<td><input type="number" step="0.1" value="${Number((v*(k<dim?1000:180/Math.PI)).toPrecision(6))}" aria-label="${names[k]} punto ${i}"></td>`).join('')}<td><button class="remove-path" data-index="${i}" aria-label="Eliminar punto ${i}">×</button></td></tr>`).join('');
    for (const id of ['nodeRows', 'pathRows', 'beamRows']) $(id).onchange = () => {
        readRows();
        if (id === 'nodeRows') drawEditor();
    };
    $('restFactor').oninput = readRows;
    document.querySelectorAll('.remove-node').forEach(b => b.onclick = () => {
        readRows();
        const i = Number(b.dataset.index);
        draft.positions.splice(i, 1);
        draft.roles.splice(i, 1);
        const edges = [],
            radii = [];
        draft.edges.forEach((e, k) => {
            if (!e.includes(i)) {
                edges.push(e.map(x => x > i ? x - 1 : x));
                radii.push(draft.radii[k]);
            }
        });
        draft.edges = edges;
        draft.radii = radii;
        drawEditor();
    });
    document.querySelectorAll('.remove-path').forEach(b => b.onclick = () => {
        readRows();
        const i = Number(b.dataset.index);
        draft.factors.splice(i, 1);
        draft.poses.splice(i, 1);
        drawEditor();
    });
    preview();
}

function interpolateDraft(t) {
    const f = draft.factors,
        p = draft.poses;
    if (f.length < 2) return p[0] || [];
    let i = 0;
    while (i < f.length - 2 && t >= f[i + 1]) i++;
    const h = f[i + 1] - f[i];
    if (h <= 0) return p[i];
    const u = Math.min(1, Math.max(0, (t - f[i]) / h));
    const slopes = p.slice(1).map((q, j) => q.map((v, k) => (v - p[j][k]) / (f[j + 1] - f[j])));
    const tangent = j => j === 0 ? slopes[0] : j === p.length - 1 ? slopes.at(-1) : slopes[j].map((v, k) => (v + slopes[j - 1][k]) / 2);
    const a = tangent(i),
        b = tangent(i + 1);
    return p[i].map((v, k) => (2 * u ** 3 - 3 * u ** 2 + 1) * v + (u ** 3 - 2 * u ** 2 + u) * h * a[k] + (-2 * u ** 3 + 3 * u ** 2) * p[i + 1][k] + (u ** 3 - u ** 2) * h * b[k]);
}

function preview() {
    if (!draft?.positions.length) return;
    const p = draft.positions,
        dim = draft.dimension,
        mobile = draft.roles.map((r, i) => r === 'mobile' ? i : -1).filter(i => i >= 0);
    const c = Array(dim).fill(0).map((_, k) => mobile.reduce((s, i) => s + p[i][k], 0) / Math.max(1, mobile.length));
    const project = p => dim === 2 ? [p[0], p[1]] : [p[0] - .4 * p[2], p[1] + .25 * p[2]];
    // Preview uses true physical displacements and an independent fitted camera.
    const targets = draft.poses.map(q => project(c.map((v, k) => v + q[k]))),
        all = [...p.map(project), ...targets];
    const lo = [0, 1].map(k => Math.min(...all.map(a => a[k]))),
        hi = [0, 1].map(k => Math.max(...all.map(a => a[k]))),
        scale = Math.min(500 / Math.max(hi[0] - lo[0], .02), 300 / Math.max(hi[1] - lo[1], .02));
    const screen = p => [300 + (p[0] - (lo[0] + hi[0]) / 2) * scale, 210 - (p[1] - (lo[1] + hi[1]) / 2) * scale];
    const xy = p => screen(project(p));
    let svg = '';
    draft.edges.forEach(([i, j], k) => {
        if (!p[i] || !p[j]) return;
        const a = xy(p[i]),
            b = xy(p[j]);
        svg += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#6d9fa9" stroke-width="${Math.max(1,draft.radii[k]*2000)}"/>`;
    });
    const path = targets.map(screen);
    svg += `<polyline points="${path.map(v=>v.join(',')).join(' ')}" fill="none" stroke="#eab4fa" stroke-width="2"/>`;
    path.forEach((a, i) => {
        svg += `<circle cx="${a[0]}" cy="${a[1]}" r="3" fill="#eab4fa"/><text x="${a[0]+5}" y="${a[1]-5}" fill="#eab4fa" font-size="10">${i}</text>`;
    });
    p.forEach((v, i) => {
        const a = xy(v),
            color = {
                fixed: '#f19a90',
                mobile: '#6fe0ba',
                free: '#74b9ef'
            } [draft.roles[i]];
        svg += `<circle cx="${a[0]}" cy="${a[1]}" r="5" fill="${color}"/><text x="${a[0]+8}" y="${a[1]+12}" fill="#bbcbd5" font-size="11">${i}</text>`;
    });
    if (mobile.length >= 2) {
        const a = xy(p[mobile[0]]),
            b = xy(p[mobile[1]]);
        svg += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#d8f4e9" stroke-width="5"/>`;
    }
    $('specPreview').innerHTML = svg;
    const samples = Array.from({
        length: 101
    }, (_, i) => interpolateDraft(i / 100));
    if (samples.length && samples[0].length) {
        const xx = samples.map(q => q[0]),
            yy = samples.map(q => q[1]);
        const minX = Math.min(...xx),
            maxX = Math.max(...xx),
            minY = Math.min(...yy),
            maxY = Math.max(...yy);
        const fit = Math.min(480 / Math.max(maxX - minX, .001), 150 / Math.max(maxY - minY, .001));
        const at = q => [300 + (q[0] - (minX + maxX) / 2) * fit, 110 - (q[1] - (minY + maxY) / 2) * fit];
        let detail = `<polyline points="${samples.map(q=>at(q).join(',')).join(' ')}" fill="none" stroke="#eab4fa" stroke-width="2"/>`;
        draft.poses.forEach((q, i) => {
            const v = at(q),
                angle = q.at(-1);
            detail += `<circle cx="${v[0]}" cy="${v[1]}" r="4" fill="#eab4fa"/><text x="${v[0]+8}" y="${v[1]-8}" fill="#bbcbd5" font-size="11">${i}</text>`;
            if (dim === 2) detail += `<line x1="${v[0]}" y1="${v[1]}" x2="${v[0]+22*Math.cos(angle)}" y2="${v[1]-22*Math.sin(angle)}" stroke="#91e4c8" stroke-width="2"/>`;
        });
        $('pathPreview').innerHTML = detail;
    }
    $('beamCount').textContent = '· ' + draft.edges.length;
    $('specSummary').textContent = `${dim}D · ${p.length} nodos · ${draft.edges.length} vigas · ${draft.poses.length} poses objetivo`;
}
$('navSpec').onclick = $('editSpec').onclick = () => screen('spec');
$('navSynthesis').onclick = () => {
    if (data) screen('synthesis');
};
$('dimension').onchange = () => {
    readRows();
    const dim = Number($('dimension').value);
    draft.dimension = dim;
    draft.positions = draft.positions.map(p => dim === 3 ? [...p, 0] : p.slice(0, 2));
    draft.poses = draft.poses.map(q => dim === 3 ? [q[0], q[1], 0, 0, 0, q[2]] : [q[0], q[1], q[5]]);
    drawEditor();
};
$('addNode').onclick = () => {
    readRows();
    draft.positions.push(Array(draft.dimension).fill(0).map((_, k) => k === 0 ? .06 : 0));
    draft.roles.push('free');
    drawEditor();
};
$('addWaypoint').onclick = () => {
    readRows();
    let largest = 0,
        index = 0;
    for (let i = 0; i < draft.factors.length - 1; i++) {
        const gap = draft.factors[i + 1] - draft.factors[i];
        if (gap > largest) {
            largest = gap;
            index = i;
        }
    }
    if (!largest) return;
    draft.factors.splice(index + 1, 0, (draft.factors[index] + draft.factors[index + 1]) / 2);
    draft.poses.splice(index + 1, 0, draft.poses[index].map((v, k) => (v + draft.poses[index + 1][k]) / 2));
    drawEditor();
};
$('defaults').onclick = async () => {
    draft = null;
    await action('/api/reset', {
        dimension: Number($('dimension').value)
    });
};
$('reset').onclick = () => action('/api/restart', {});
$('startSynthesis').onclick = async () => {
    if (busy || !draft) return;
    readRows();
    $('specError').textContent = '';
    const body = {
        design: draft,
        factors: draft.factors,
        poses: draft.poses,
        rest_factor: draft.rest_factor,
    };
    const oldData = data;
    await action('/api/specification', body);
    if (data !== oldData) {
        $('specDigest').textContent = $('specSummary').textContent;
        screen('synthesis');
    } else $('specError').textContent = $('status').textContent;
};

const parameterGroups=[
 ['Mutaciones',[['mutation_scale','Intensidad de mutación',0.001,1,.01],['geometry_mutation_probability','Prob. mover nodo',0,1,.05],['radius_mutation_probability','Prob. cambiar radio',0,1,.05],['presence_mutation_probability','Prob. cambiar peso de viga',0,1,.05],['topology_mutation_probability','Prob. añadir/eliminar viga',0,1,.01],['geometry_scale','Escala de geometría (m)',.0001,.1,.001]]],
 ['Selección y diversidad',[['diversity_weight','Peso de diversidad (0 = desactivar)',0,5,.05],['diversity_radius','Radio de similitud',.01,2,.05],['diversity_geometry_weight','Similitud de geometría',0,5,.1],['diversity_topology_weight','Similitud de conexiones',0,5,.1],['diversity_radius_weight','Similitud de radios',0,5,.1],['diversity_presence_weight','Similitud de pesos de viga',0,5,.1],['focused_fraction','Cuota de variantes locales',0,1,.05],['parent_pool_fraction','Fracción de mejores padres',.01,1,.05],['local_mutation_factor','Intensidad local relativa',.01,1,.05],['crossover_probability','Prob. de cruce exploratorio',0,1,.05],['tournament_size','Tamaño del torneo',1,32,1]]],
 ['Consigna y coste',[['position_weight','Peso de posición',0,20,.1],['orientation_weight','Peso de orientación',0,20,.1],['position_scale','Escala de posición (m)',.00001,1,.001],['orientation_scale','Escala de orientación (rad)',.0001,3,.01],['material_weight','Peso de material',0,10,.01],['mode_separation_weight','Peso de separación de rigidez',0,10,.01],['transverse_stiffness_weight','Peso de rigidez transversal',0,10,.01],['minimum_transverse_stiffness','Rigidez transversal mínima (N/m)',1,1e9,100]]],
 ['Ciclo y gradiente',[['gradient_steps_per_evolution','Pasos de gradiente antes de evolucionar',0,100,1],['evolution_steps_per_cycle','Generaciones por ciclo',1,100,1],['learning_rate','Tamaño del paso',.0001,1,.005],['gradient_geometry_rate','Paso relativo de geometría',0,10,.1],['gradient_radius_rate','Paso relativo de radios',0,10,.1],['gradient_presence_rate','Paso relativo de pesos de viga',0,10,.1],['gradient_backtracks','Máximo de intentos de paso',1,16,1],['evaluation_batch_size','Individuos por lote',1,128,1]]]
];
const continuationFields=[['steps','Incrementos de recorrido natural',1,128,1],['characteristic_length','Longitud característica (m)',.001,1,.005],['mode_gap_tolerance','Separación mínima relativa de modos',1e-9,.1,.000001],['max_iterations','Máximo Newton',1,120,1],['hessian_chunk_size','Filas de tangente por bloque',1,32,1]];
function parameterFields(fields, config, attribute='parameter') {
 return `<div class="parameter-grid">${fields.map(([key,label,min,max,step])=>`<label>${label}<input data-${attribute}="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${config[key]}"></label>`).join('')}</div>`;
}
let activeSettingsTab=0;
function selectSettingsTab(index, focus=false) {
 activeSettingsTab=index;
 $('settingsTabs').querySelectorAll('[role=tab]').forEach((tab,i)=>{
  const selected=i===index;
  tab.setAttribute('aria-selected',String(selected));
  tab.tabIndex=selected?0:-1;
  $(tab.getAttribute('aria-controls')).hidden=!selected;
  if(selected && focus)tab.focus();
 });
}
$('settingsTabs').querySelectorAll('[role=tab]').forEach((tab,i)=>{
 tab.onclick=()=>selectSettingsTab(i);
 tab.onkeydown=e=>{
  const count=$('settingsTabs').children.length;
  let next;
  if(['ArrowRight','ArrowDown'].includes(e.key))next=(i+1)%count;
  else if(['ArrowLeft','ArrowUp'].includes(e.key))next=(i+count-1)%count;
  else if(e.key==='Home')next=0;
  else if(e.key==='End')next=count-1;
  else return;
  e.preventDefault();selectSettingsTab(next,true);
 };
});
function readParameters(scope,attribute){return Object.fromEntries([...scope.querySelectorAll(`[data-${attribute}]`)].map(input=>[input.dataset[attribute],Number(input.value)]));}
function readContinuation(scope){return readParameters(scope,'continuation');}
let liveConfigSignature='';
window.syncLiveSearch=function(state){
 const signature=JSON.stringify([state.search_config,state.continuation_config]);
 if(signature===liveConfigSignature)return;
 liveConfigSignature=signature;
 const groups=[['Población',[['population','Estructuras retenidas',2,256,1],['elites','Élites protegidas',1,255,1]]],parameterGroups[3],parameterGroups[0],parameterGroups[1],parameterGroups[2]];
 $('liveParameters').innerHTML=groups.map(([title,fields],i)=>`<section id="settingsPanel${i}" class="synthesis-setting-panel" role="tabpanel" aria-labelledby="settingsTab${i}" hidden><h3>${title}</h3>${parameterFields(fields,state.search_config)}</section>`).join('');
 $('computeDevice').value=state.search_config.device;
 $('liveContinuation').innerHTML='<h3>Continuación FEM</h3>'+parameterFields(continuationFields,state.continuation_config,'continuation');
 selectSettingsTab(activeSettingsTab);
};
$('openSearchSettings').onclick=()=>{
 running=false;playing=false;
 $('run').textContent='Iniciar búsqueda';$('play').textContent='Animar';
 $('searchSettingsError').textContent='';
 $('searchSettings').showModal();
};
$('closeSearchSettings').onclick=()=>$('searchSettings').close();
$('searchSettings').addEventListener('close',()=>requestAnimationFrame(render));
async function applySynthesisSettings(restart){
 if(busy)return;
 $('searchSettingsError').textContent='';
 const device=$('computeDevice').value;
 if(!restart && device!==data.search_config.device){
  $('searchSettingsError').textContent='Para cambiar el dispositivo, usa «Aplicar y reiniciar».';
  return;
 }
 const previous=data;
 const search=readParameters($('liveParameters'),'parameter');
 if(restart)search.device=device;
 await action(restart?'/api/restart':'/api/search-settings',{search,continuation:readContinuation($('liveContinuation'))});
 if(data!==previous)$('searchSettings').close();
 else $('searchSettingsError').textContent=$('status').textContent;
}
$('applySearch').onclick=()=>applySynthesisSettings(false);
$('restartSearch').onclick=()=>applySynthesisSettings(true);
$('bidirectionalExample').onclick=()=>{
 readRows();
 const end=[...draft.poses.at(-1)];
 draft.rest_factor=.5;
 $('restFactor').value=.5;
 draft.factors=[0,.15,.35,.5,.65,.85,1];
 draft.poses=draft.factors.map(t=>end.map(v=>v*(2*t-1)));
 drawEditor();
};
