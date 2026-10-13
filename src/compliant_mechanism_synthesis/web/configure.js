/* Specification editor keeps a local draft until validation succeeds. */
let draft = null;

function namesFor(dim) {
    return ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz'];
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
    draft.inputPoses=structuredClone(state.input_waypoints.poses);
    draft.rest_factor = state.waypoints.rest_factor;
    $("restFactor").value=draft.rest_factor;
    draft.radiusBounds = state.search_config;
    $('initialNeighbours').value = state.search_config.initial_neighbours;
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
    draft.inputPoses = [];
    document.querySelectorAll('#pathRows tr').forEach(row => {
        const values = [...row.querySelectorAll('input')].map(input => Number(input.value));
        draft.factors.push(values[0]);
        const dof=6;
        const convert=(v,k)=>k<draft.dimension?v/1000:v*Math.PI/180;
        draft.inputPoses.push(values.slice(1,1+dof).map(convert));
        draft.poses.push(values.slice(1+dof).map(convert));
    });
    draft.rest_factor = Number($('restFactor').value);
    preview();
}

function drawEditor() {
    const dim = draft.dimension,
        axes = ['X', 'Y', 'Z'].slice(0, dim),
        names = namesFor(dim);
    $('nodeHead').innerHTML = '<tr><th>Nodo</th>' + axes.map(a => `<th>${a} (m)</th>`).join('') + '<th>Rol</th><th></th></tr>';
    $('nodeRows').innerHTML = draft.positions.map((p, i) => `<tr><td>${i}</td>${p.map(v=>`<td><input type="number" step="0.001" value="${v}" aria-label="Nodo ${i} coordenada"></td>`).join('')}<td><select aria-label="Rol nodo ${i}">${['fixed','active','passive','free'].map((r,k)=>`<option value="${r}" ${draft.roles[i]===r?'selected':''}>${['Fijo','Activo','Pasivo','Libre'][k]}</option>`).join('')}</select></td><td><button class="remove-node" data-index="${i}" aria-label="Eliminar nodo ${i}">×</button></td></tr>`).join('');
    const enabled = new Map(draft.edges.map((e, i) => [
        [...e].sort((a, b) => a - b).join(','), draft.radii[i] * 1000
    ]));
    let beams = '';
    for (let i = 0; i < draft.positions.length; i++)
        for (let j = i + 1; j < draft.positions.length; j++) {
            if (draft.roles[i] === draft.roles[j] && ['fixed', 'active', 'passive'].includes(draft.roles[i])) continue;
            const key = [i, j].join(','),
                radius = enabled.get(key);
            beams += `<label class="beam-row" data-edge="${key}"><input type="checkbox" ${radius!==undefined?'checked':''}>${i}–${j}<input type="number" min="${draft.radiusBounds.radius_min*1000+.001}" max="${draft.radiusBounds.radius_max*1000-.001}" step="0.05" value="${radius??draft.radiusBounds.initial_radius*1000}" aria-label="Radio viga ${i} ${j}"></label>`;
        }
    $('beamRows').innerHTML = beams;
    $('pathHead').innerHTML = '<tr><th>Progreso 0–1</th>' + ['Activo','Pasivo'].flatMap(body=>names.map((n,k)=>`<th>${body} ${n} (${k<dim?'mm':'°'})</th>`)).join('') + '<th></th></tr>';
    $('pathRows').innerHTML = draft.factors.map((t, i) => `<tr><td><input type="number" min="0" max="1" step="0.05" value="${t}" aria-label="Progreso punto ${i}"></td>${[...draft.inputPoses[i],...draft.poses[i]].map((v,k)=>`<td><input type="number" step="0.1" value="${Number((v*(k%names.length<dim?1000:180/Math.PI)).toPrecision(6))}" aria-label="${k<names.length?"Activo":"Pasivo"} ${names[k%names.length]} punto ${i}"></td>`).join('')}<td><button class="remove-path" data-index="${i}" aria-label="Eliminar punto ${i}">×</button></td></tr>`).join('');
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
        draft.inputPoses.splice(i,1);
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

let previewYaw=.5,previewPitch=.32,previewOrbitFrame=null,previewOrbitTime=null;
function previewProjection(p){
 const a=Math.cos(previewYaw)*p[0]-Math.sin(previewYaw)*p[2];
 const b=Math.sin(previewYaw)*p[0]+Math.cos(previewYaw)*p[2];
 return [a,Math.cos(previewPitch)*p[1]-Math.sin(previewPitch)*b];
}
function preview() {
    if (!draft?.positions.length) return;
    const p = draft.positions,
        dim = draft.dimension,
        mobile = draft.roles.map((r, i) => r === 'passive' ? i : -1).filter(i => i >= 0);
    const c = Array(dim).fill(0).map((_, k) => mobile.reduce((s, i) => s + p[i][k], 0) / Math.max(1, mobile.length));
    const project = previewProjection;
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
                fixed: '#f19a90',active:'#f5ad68',
                passive: '#6fe0ba', active: '#f5ad68',
                free: '#74b9ef'
            } [draft.roles[i]];
        svg += `<circle cx="${a[0]}" cy="${a[1]}" r="5" fill="${color}"/><text x="${a[0]+8}" y="${a[1]+12}" fill="#bbcbd5" font-size="11">${i}</text>`;
    });
    for (const [role,color] of [['fixed','#f19a90'],['active','#f5ad68'],['passive','#6fe0ba']]) {
        const ids = draft.roles.flatMap((r,i)=>r===role?[i]:[]);
        for(let j=0;j<ids.length;j++) for(let k=j+1;k<ids.length;k++) {
            const a=xy(p[ids[j]]),b=xy(p[ids[k]]);
            svg += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="3"/>`;
        }
    }
    $('specPreview').innerHTML = svg;
    const samples = Array.from({
        length: 101
    }, (_, i) => interpolateDraft(i / 100));
    if (samples.length && samples[0].length) {
        const projected=samples.map(q=>project(q.slice(0,3)));
        const minX=Math.min(...projected.map(p=>p[0])),maxX=Math.max(...projected.map(p=>p[0]));
        const minY=Math.min(...projected.map(p=>p[1])),maxY=Math.max(...projected.map(p=>p[1]));
        const fit=Math.min(480/Math.max(maxX-minX,.001),150/Math.max(maxY-minY,.001));
        const at=p=>[300+(p[0]-(minX+maxX)/2)*fit,110-(p[1]-(minY+maxY)/2)*fit];
        const axesLength=Math.max(.001,maxX-minX,maxY-minY)*.1;
        let detail=`<polyline points="${projected.map(p=>at(p).join(',')).join(' ')}" fill="none" stroke="#eab4fa" stroke-width="2"/>`;
        draft.poses.forEach((q,i)=>{
            const origin=q.slice(0,3),a=at(project(origin));
            detail+=`<circle cx="${a[0]}" cy="${a[1]}" r="4" fill="#eab4fa"/><text x="${a[0]+8}" y="${a[1]-8}" fill="#bbcbd5" font-size="11">${i}</text>`;
            for(let k=0;k<3;k++){
                const vector=rotate([0,1,2].map(j=>j===k?axesLength:0),q.slice(3));
                const b=at(project(add(origin,vector)));
                detail+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#eab4fa" stroke-width="1.2"/>`;
            }
        });
        $('pathPreview').innerHTML = detail;
    }
    $('beamCount').textContent = '· ' + draft.edges.length;
    $('specSummary').textContent = `${dim}D · ${p.length} nodos · ${draft.edges.length} vigas · ${draft.poses.length} poses objetivo`;
}
$('navSpec').onclick = () => screen('spec');
$('navSynthesis').onclick = () => {
    if (data) screen('synthesis');
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
    for(const key of ["poses","inputPoses"])draft[key].splice(index+1,0,draft[key][index].map((v,k)=>(v+draft[key][index+1][k])/2));
    drawEditor();
};
$('defaults').onclick = async () => {
    draft = null;
    await action('/api/reset', {
        initial_neighbours: Number($('initialNeighbours').value)
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
        input_poses: draft.inputPoses,
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
 ['Mutaciones',[['mutation_scale','Intensidad de mutación',0.001,1,.01],['geometry_mutation_probability','Prob. mover nodo',0,1,.05],['radius_mutation_probability','Prob. cambiar radio',0,1,.05],['topology_mutation_probability','Prob. de aplicar cada edición',0,1,.05],['topology_mutation_edits','Máximo de conexiones editadas',1,16,1],['topology_add_probability','Prob. añadir frente a eliminar',0,1,.05],['topology_distance_scale','Distancia de referencia de conexiones (m)',.001,.3,.005],['topology_distance_power','Intensidad del sesgo por distancia (0 = uniforme)',0,8,.5],['geometry_scale','Escala de geometría (m)',.0001,.1,.001]]],
 ['Selección y diversidad',[['diversity_weight','Peso de diversidad (0 = desactivar)',0,5,.05],['diversity_radius','Radio de similitud',.01,2,.05],['diversity_geometry_weight','Similitud de geometría',0,5,.1],['diversity_topology_weight','Similitud de conexiones',0,5,.1],['diversity_radius_weight','Similitud de radios',0,5,.1],['focused_fraction','Cuota de variantes locales',0,1,.05],['parent_pool_fraction','Fracción de mejores padres',.01,1,.05],['local_mutation_factor','Intensidad local relativa',.01,1,.05],['crossover_probability','Prob. de cruce exploratorio',0,1,.05],['tournament_size','Tamaño del torneo',1,32,1]]],
 ['Consigna y coste',[['position_weight','Peso de posición',0,20,.1],['orientation_weight','Peso de orientación',0,20,.1],['position_scale','Escala de posición (m)',.00001,1,.001],['orientation_scale','Escala de orientación (rad)',.0001,3,.01],['material_weight','Peso de material',0,10,.01],['passive_support_weight','Peso de soporte mínimo del pasivo',0,10,.01],['minimum_passive_support','Soporte mínimo · todas las direcciones (N/m)',1,1e9,100]]],
 ['Ciclo y gradiente',[['gradient_steps_per_evolution','Pasos de gradiente antes de evolucionar',0,100,1],['offspring_gradient_steps','Pasos de gradiente por descendiente antes de seleccionar',1,20,1],['learning_rate','Tasa inicial / manual',.000001,1,.005],['adaptive_learning_rate','Modo de la tasa',0,1,1],['gradient_monotonic_target','Fracción que debe mejorar en el primer intento',.1,1,.01],['learning_rate_adaptation','Respuesta de la adaptación',.1,10,.1],['learning_rate_min','Tasa mínima',.000001,1,.00001],['learning_rate_max','Tasa máxima',.000001,1,.01],['gradient_geometry_rate','Paso relativo de geometría',0,10,.1],['gradient_radius_rate','Paso relativo de radios',0,10,.1],['gradient_backtracks','Máximo de intentos de paso',1,16,1],['topology_batch_union_ratio','Máximo de vigas calculadas / utilizadas por lote',1,8,.25],['evaluation_batch_size','Individuos por lote de evaluación',1,128,1],['gradient_batch_size','Individuos por lote de gradiente',1,128,1]]],
 ['Geometría',[['collision_weight','Peso de colisiones',0,20,.05],['collision_clearance','Separación adicional (m)',0,.02,.0001],['collision_subdivisions','Segmentos para comprobar cada viga',1,16,1],['collision_path_substeps','Muestras por intervalo del recorrido',1,8,1],['collision_joint_allowance','Zona de unión permitida (× suma de radios)',1,5,.25],['beam_length_weight','Peso de longitudes',0,20,.05],['beam_length_min','Longitud mínima preferida (m)',.001,.15,.001],['beam_length_max','Longitud máxima preferida (m)',.002,.3,.001]]]
];
const continuationFields=[['beam_segments','Segmentos internos por viga',1,8,1],['predictor','Predictor de equilibrio (0 = desactivar)',0,1,1],['path_position_tolerance','Corrección máxima del predictor · posición (m)',.000001,.01,.00001],['path_orientation_tolerance','Corrección máxima del predictor · orientación (rad)',.0001,.2,.001],['steps','Incrementos de entrada',1,128,1],['characteristic_length','Longitud característica (m)',.001,1,.005],['max_iterations','Máximo Newton',1,120,1],['hessian_chunk_size','Filas de tangente por bloque',1,32,1]];
function parameterFields(fields, config, attribute='parameter') {
 const controls=fields.map(([key,label,min,max,step])=>{
  const control=key==='adaptive_learning_rate'
   ? `<select data-parameter="${key}"><option value="1" ${config[key]===1?'selected':''}>Adaptativa</option><option value="0" ${config[key]===0?'selected':''}>Manual</option></select>`
   : `<input data-${attribute}="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${config[key]}">`;
  return `<label><span class="field-label">${label}</span>${control}</label>`;
 });
 return `<div class="parameter-grid">${controls.join('')}</div>`;
}
const settingSections={
 costSettings:[['Movimiento',['position_weight','orientation_weight','position_scale','orientation_scale']],['Material y soporte',['material_weight','passive_support_weight','minimum_passive_support']]],
 gradientSettings:[['Tamaño de paso',['learning_rate']],['Adaptación',['adaptive_learning_rate','gradient_monotonic_target','learning_rate_adaptation','learning_rate_min','learning_rate_max']],['Variables y evaluación',['gradient_geometry_rate','gradient_radius_rate','gradient_backtracks','gradient_batch_size']]],
 evolutionSettings:[['Qué cambia',['geometry_mutation_probability','radius_mutation_probability','topology_mutation_probability','topology_mutation_edits','topology_add_probability']],['Alcance del cambio',['mutation_scale','geometry_scale','topology_distance_scale','topology_distance_power']],['Padres y descendientes',['focused_fraction','parent_pool_fraction','local_mutation_factor','crossover_probability','tournament_size','offspring_gradient_steps']]],
 analysisSettings:[['Malla y recorrido',['beam_segments','steps','predictor']],['Precisión del predictor',['path_position_tolerance','path_orientation_tolerance']],['Equilibrio',['max_iterations','characteristic_length','hessian_chunk_size']]]
};
function sectionFields(id,fields,config,attribute){
 const groups=settingSections[id];
 if(!groups)return parameterFields(fields,config,attribute);
 return `<div class="setting-groups">${groups.map(([title,keys])=>`<fieldset><legend>${title}</legend>${parameterFields(keys.map(key=>fields.find(f=>f[0]===key)),config,attribute)}</fieldset>`).join('')}</div>`;
}
function readParameters(scope,attribute){return Object.fromEntries([...scope.querySelectorAll(`[data-${attribute}]`)].map(input=>[input.dataset[attribute],input.tagName==='SELECT' && input.dataset[attribute]!=='adaptive_learning_rate'?input.value:Number(input.value)]));}
const field = key => parameterGroups.flatMap(([,fields])=>fields).find(f=>f[0]===key);
const localGroups=[
 ['cycleSettings','Ciclo de búsqueda',['gradient_steps_per_evolution'].map(field)],
 ['populationSettings','Tamaño de población',[['population','Estructuras retenidas',2,256,1],['elites','Élites protegidas',1,255,1],field('evaluation_batch_size'),field('topology_batch_union_ratio')]],
 ['gradientSettings','Paso y refinamiento',['learning_rate','adaptive_learning_rate','gradient_monotonic_target','learning_rate_adaptation','learning_rate_min','learning_rate_max','gradient_geometry_rate','gradient_radius_rate','gradient_backtracks','gradient_batch_size'].map(field)],
 ['evolutionSettings','Variación y selección',[...parameterGroups[0][1],...['focused_fraction','parent_pool_fraction','local_mutation_factor','crossover_probability','tournament_size','offspring_gradient_steps'].map(field)]],
 ['diversitySettings','Diversidad',parameterGroups[1][1].filter(f=>f[0].startsWith('diversity_'))],
 ['costSettings','Criterio de ajuste',parameterGroups[2][1]],
 ['analysisSettings','Resolución FEM',continuationFields,'continuation'],
 ['collisionSettings','Colisiones',['collision_weight','collision_clearance','collision_subdivisions','collision_path_substeps','collision_joint_allowance'].map(field)],
 ['beamLengthSettings','Longitudes',['beam_length_weight','beam_length_min','beam_length_max'].map(field)],
 ['computeSettings','Dispositivo',[]]
];
const localSignatures=new Map();
window.syncLiveSearch=function(state){
 for(const [id,title,fields,attribute='parameter'] of localGroups){
  const config=attribute==='continuation'?state.continuation_config:state.search_config;
  const signature=JSON.stringify([fields.map(([key])=>config[key]),id==='analysisSettings'?[state.dimension,config.newton_batch_mode]:null,id==='computeSettings'?config.device:null]);
  if(localSignatures.get(id)===signature)continue;
  localSignatures.set(id,signature);
  const scope=$(id);
  delete scope.dataset.dirty;
  let content=sectionFields(id,fields,config,attribute);
  if(id==='analysisSettings')content+=`<label>Lotes de Newton<select data-continuation="newton_batch_mode">${[['auto','Auto · según dispositivo'],['compact','Compactar diseños pendientes'],['masked','Lote fijo con máscaras']].map(([value,label])=>`<option value="${value}" ${config.newton_batch_mode===value?'selected':''}>${label}</option>`).join('')}</select></label>`;
  if(id==='computeSettings')content=`<label>Dispositivo<select data-parameter="device">${[['cpu','CPU'],['xpu','Intel XPU'],['auto','Auto · medir CPU y XPU']].map(([value,label])=>`<option value="${value}" ${config.device===value?'selected':''}>${label}</option>`).join('')}</select></label><p class="note">Reinicia la población desde la especificación aplicada. CPU suele ser más rápida con lotes pequeños.</p>`;
  scope.setAttribute('aria-label',title);
  scope.innerHTML=`${settingSections[id]?'':`<h3>${title}</h3>`}${content}<div class="setting-actions"><button class="secondary" data-apply-settings="${id}">${id==='computeSettings'?'Aplicar y reiniciar':'Aplicar cambios'}</button><span class="setting-feedback" role="status"></span></div>`;
  if(id==='analysisSettings')scope.querySelectorAll('fieldset')[2].querySelector('.parameter-grid').append(scope.querySelector('select').closest('label'));
  scope.querySelector('button').onclick=()=>applyLocalSettings(id);
  scope.querySelectorAll('input,select').forEach(input=>input.addEventListener('input',()=>{scope.dataset.dirty='true';scope.querySelector('.setting-feedback').textContent='Cambios sin aplicar';}));
  if(id==='analysisSettings'){
   const segments=scope.querySelector('[data-continuation="beam_segments"]'),mode=scope.querySelector('[data-continuation="newton_batch_mode"]');

   const update=()=>mode.disabled=Number(segments.value)>1;segments.addEventListener('input',update);update();
  }
 }
};
async function applyLocalSettings(id){
 if(busy)return;
 const scope=$(id),inputs=[...scope.querySelectorAll('input')];
 if(inputs.some(input=>input.value==='' || !Number.isFinite(Number(input.value)) || Number(input.value)<Number(input.min) || Number(input.value)>Number(input.max))){scope.querySelector('.setting-feedback').textContent='Revisa los valores y sus límites.';return;}
 running=false;$('run').textContent='Iniciar búsqueda';
 const previous=data;
 const body={search:readParameters(scope,'parameter'),continuation:readParameters(scope,'continuation')};
 await action(id==='computeSettings'?'/api/restart':'/api/search-settings',body);
 $(id).querySelector('.setting-feedback').textContent=data!==previous?'Aplicado':$('status').textContent;
 if(data!==previous)delete $(id).dataset.dirty;
}
$('bidirectionalExample').onclick=()=>{
 readRows();
 const end=[...draft.poses.at(-1)];
 const inputEnd=[...draft.inputPoses.at(-1)];
 draft.rest_factor=.5;
 $('restFactor').value=.5;
 draft.factors=[0,.15,.35,.5,.65,.85,1];
 draft.poses=draft.factors.map(t=>end.map(v=>v*(2*t-1)));
 draft.inputPoses=draft.factors.map(t=>inputEnd.map(v=>v*(2*t-1)));
 drawEditor();
};

function previewOrbit(time){
 if($('previewAutoRotate').getAttribute('aria-pressed')!=='true'){previewOrbitFrame=null;return;}
 const box=$('specPreview').getBoundingClientRect();
 if(!draft || $('specScreen').hidden || document.hidden || box.bottom<0 || box.top>innerHeight)previewOrbitTime=null;
 else if(previewOrbitTime===null)previewOrbitTime=time;
 else if(time-previewOrbitTime>=1000/30){previewYaw+=Math.min(100,time-previewOrbitTime)*Math.PI/9000;previewOrbitTime=time;preview();}
 previewOrbitFrame=requestAnimationFrame(previewOrbit);
}
$('previewAutoRotate').onclick=()=>{
 const enabled=$('previewAutoRotate').getAttribute('aria-pressed')!=='true';
 $('previewAutoRotate').setAttribute('aria-pressed',String(enabled));
 $('previewAutoRotate').textContent=enabled?'Detener rotación':'Rotación automática';
 previewOrbitTime=null;
 if(previewOrbitFrame!==null)cancelAnimationFrame(previewOrbitFrame);
 previewOrbitFrame=enabled?requestAnimationFrame(previewOrbit):null;
};
