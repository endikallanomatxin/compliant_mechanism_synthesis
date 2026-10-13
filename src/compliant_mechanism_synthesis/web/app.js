const $ = id => document.getElementById(id);
let data, scene, busy = false,
    running = false,
    playing = false,
    automaticScale = true,
    yaw = .45,
    pitch = .3;
const number = x => x === null ? '—' : Number(x).toPrecision(4);
const add = (a, b) => a.map((v, k) => v + b[k]);
const sub = (a, b) => a.map((v, k) => v - b[k]);
const mul = (a, f) => a.map(v => v * f);
const magnitude = a => Math.hypot(...a);
const axisColors = ['#f19a90', '#6fe0ba', '#74b9ef'];

function rotate(v, r) {
    const angle = magnitude(r);
    if (angle < 1e-12) return [...v];
    const n = mul(r, 1 / angle),
        c = Math.cos(angle),
        s = Math.sin(angle),
        dot = n.reduce((a, x, k) => a + x * v[k], 0);
    const cross = [n[1] * v[2] - n[2] * v[1], n[2] * v[0] - n[0] * v[2], n[0] * v[1] - n[1] * v[0]];
    return v.map((x, k) => x * c + cross[k] * s + n[k] * dot * (1 - c));
}
const mobileIds = () => data.roles.map((r, i) => r === 'passive' ? i : -1).filter(i => i >= 0);

function center(positions,role='passive') {
    const ids = data.roles.map((r,i)=>r===role?i:-1).filter(i=>i>=0);
    return positions[0].map((_, k) => ids.reduce((s, i) => s + positions[i][k], 0) / ids.length);
}

function project(p) {
    const x = p[0] - .06,
        y = p[1],
        z = p[2],
        a = Math.cos(yaw) * x - Math.sin(yaw) * z,
        b = Math.sin(yaw) * x + Math.cos(yaw) * z;
    return [450 + a * 4300, 300 - (Math.cos(pitch) * y - Math.sin(pitch) * b) * 4300];
}

function displayPositions(state,scale) {
 const centers={};for(const role of ['active','passive']){
  const c0=center(data.positions,role),c=center(state.positions,role);
  centers[role]={c,visible:add(c0,mul(sub(c,c0),scale))};
 }
 return state.positions.map((p,i)=>centers[data.roles[i]]?add(centers[data.roles[i]].visible,sub(p,centers[data.roles[i]].c)):add(data.positions[i],mul(sub(p,data.positions[i]),scale)));
}

function bodyFrame(state, scale,role='passive') {
    const positions = displayPositions(state, scale),
        origin = center(positions,role),
        r = (role==='active'?state.input:state.output).slice(data.dimension);
    const axes = Array.from({
        length: data.dimension
    }, (_, k) => add(origin, rotate(Array.from({
        length: data.dimension
    }, (_, j) => j === k ? .014 : 0), r)));
    return {
        origin,
        axes
    };
}

function fitDeformation() {
    if (!data?.states.length) return;
    const span = magnitude(Array.from({
        length: data.dimension
    }, (_, k) => Math.max(...data.positions.map(p => p[k])) - Math.min(...data.positions.map(p => p[k]))));
    const maximum = Math.max(...data.target_path.map(s => magnitude(s.output.slice(0, data.dimension))), ...data.states.flatMap(s => s.positions.map((p, i) => magnitude(sub(p, data.positions[i])))));
    const factor = maximum > 1e-12 ? Math.min(10000, Math.max(1, .2 * span / maximum)) : 1;
    $('scale').value = Math.round(Math.log10(factor) * 100);
}
async function request(path, body) {
    const response = await fetch(path, body ? {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    } : {});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || 'Error de servidor');
    return result;
}
async function action(path, body) {
    if (busy) return;
    busy = true;
    document.querySelectorAll('button').forEach(b => b.disabled = b.id !== 'run');
    $('status').textContent = 'Calculando recorrido no lineal…';
    try {
        data = await request(path, body);
        window.syncSpecification?.(data);
        $('frame').max = data.states.length ? Math.round(data.states.at(-1).factor*1000) : 0;
        $('frame').value = $('frame').max;
        const names = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz'];
        if ($('coordinate').options.length !== names.length) {
            $('coordinate').innerHTML = names.map((v, i) => `<option value="${i}">${v}</option>`).join('');
            $('coordinate').value = '1';
        }
        if (automaticScale) fitDeformation();
        window.syncLiveSearch?.(data);
        window.syncNumerics?.(data);
        render();
        $('status').textContent = data.converged ? 'Recorrido convergente · carga final alcanzada' : data.message;
        if (!data.converged) running = false;
    } catch (e) {
        running = false;
        $('status').textContent = e.message;
    } finally {
        busy = false;
        document.querySelectorAll('button').forEach(b => b.disabled = false);
        $('run').textContent = running ? 'Detener búsqueda' : 'Iniciar búsqueda';
    }
    if (running) setTimeout(() => action('/api/step', {
        method: $('method').value
    }), 50);
}

function beamPoints(i, j, state, displayed, edgeIndex=null) {
    if(state.beam_nodes && edgeIndex!==null){
        const nodes=state.beam_nodes[edgeIndex], s=10**(Number($('scale').value)/100), ref=sub(data.positions[j],data.positions[i]), unit=mul(ref,1/magnitude(ref));
        const points=nodes.map((node,k)=>{const t=k/(nodes.length-1), reference=data.positions[i].map((x,d)=>x+ref[d]*t);return reference.map((x,d)=>x+(node[d]-x)*s);});
        points[0]=displayed[i]; points[points.length-1]=displayed[j];
        const result=[];
        for(let k=0;k<points.length-1;k++){
            const a=points[k], b=points[k+1], length=magnitude(sub(b,a)), ca=add(a,mul(rotate(unit,nodes[k].slice(3)),length/3)), cb=sub(b,mul(rotate(unit,nodes[k+1].slice(3)),length/3));
            for(let j=0;j<=8;j++){if(k && j===0)continue;const t=j/8,u=1-t;result.push(a.map((v,d)=>u**3*v+3*u*u*t*ca[d]+3*u*t*t*cb[d]+t**3*b[d]));}
        }
        return result;
    }
    const a = displayed[i],
        b = displayed[j],
        reference = sub(data.positions[j], data.positions[i]);
    // Cubic Hermite interpolation retains each end's section orientation.
    const length = magnitude(sub(b, a)),
        unit = mul(reference, 1 / magnitude(reference));
    const ca = add(a, mul(rotate(unit, state.rotations[i]), length / 3));
    const cb = sub(b, mul(rotate(unit, state.rotations[j]), length / 3));
    return Array.from({
        length: 25
    }, (_, k) => {
        const t = k / 24,
            u = 1 - t;
        return a.map((v, d) => u ** 3 * v + 3 * u * u * t * ca[d] + 3 * u * t * t * cb[d] + t ** 3 * b[d]);
    });
}

function render() {
    if (!data || !scene) return;
    if ($('synthesisScreen').hidden) return;
    scene.begin();
    let labels = '';
    if (!data.states.length) {
        scene.paint();
        $('labels').innerHTML = '';
        $('state').textContent = 'No hay estados de equilibrio convergentes.';

        $('poseComparison').textContent = '';
        $('orientationCompare').innerHTML = '';
        $('plot').innerHTML = '';
        drawTrajectoryOverview(null);
        return;
    }
    const state = animatedState(Number($('frame').value)/1000),
        scale = 10 ** (Number($('scale').value) / 100);
    $('scaleValue').textContent = '×' + Number(scale.toPrecision(3));
    const positions = displayPositions(state, scale),
        projected = positions.map(project),
        mobile = mobileIds();
    const text = (p, value, color = '#b6ccd8', size = 12) => {
        labels += `<text x="${p[0]}" y="${p[1]}" fill="${color}" font-size="${size}">${value}</text>`;
    };
    if ($('compare').checked) data.population.forEach(candidate => {
        if (candidate.index === data.selected_index) return;
        data.edges.forEach(([i, j], k) => {
            if (candidate.presence[k] > .05) scene.line(project(candidate.positions[i]), project(candidate.positions[j]), '#74b9ef', 1, .12);
        });
    });
    data.edges.forEach(([i, j], k) => {
        if (data.presence[k] === 0) return;
        scene.line(project(data.positions[i]), project(data.positions[j]), '#607589', 1, .45);
        scene.polyline(beamPoints(i, j, state, positions, k).map(project), '#6fe0ba', Math.max(1.5, data.radii[k] * 2600), .15 + .85 * data.presence[k]);
    });
    if ($('trajectories').checked) {
        const displayed = data.states.map(s => displayPositions(s, scale));
        scene.polyline(displayed.map(p => project(center(p))), '#6fe0ba', 3);
    }
    for(let i=0;i<mobile.length;i++)for(let j=i+1;j<mobile.length;j++)scene.line(projected[mobile[i]],projected[mobile[j]],'#6fe0ba',5);
    const fixedIds=data.roles.map((r,i)=>r==='fixed'?i:-1).filter(i=>i>=0);
    for(let i=0;i<fixedIds.length;i++)for(let j=i+1;j<fixedIds.length;j++)scene.line(projected[fixedIds[i]],projected[fixedIds[j]],'#f19a90',3);
    const activeIds=data.roles.map((r,i)=>r==='active'?i:-1).filter(i=>i>=0);
    for(let i=0;i<activeIds.length;i++)for(let j=i+1;j<activeIds.length;j++)scene.line(projected[activeIds[i]],projected[activeIds[j]],'#f5ad68',5);
    if($('trajectories').checked)scene.polyline(data.states.map(s=>project(center(displayPositions(s,scale),'active'))),'#f5ad68',2);
    projected.forEach((p, i) => {
        const color = {
            fixed: '#f19a90',
            passive: '#6fe0ba',
            active: '#f5ad68',
            free: '#74b9ef'
        } [data.roles[i]];
        if (data.roles[i] === 'fixed') {
            scene.square(p, 6, color);
            for (let h = -1; h <= 1; h++) scene.line([p[0] - 10, p[1] + h * 6], [p[0] - 17, p[1] + h * 6 + 5], color, 1);
        } else if (['active','passive'].includes(data.roles[i])) scene.square(p, 4, color);
        else scene.circle(p, 4, color);
        text([p[0] + 10, p[1] - 8], i);
    });
    if ($('frames').checked) {
        const drawFrame = (frame, title, alpha) => {
            const origin = project(frame.origin);
            scene.circle(origin, 3, '#f5cf79', alpha);
            text([origin[0] + 8, origin[1] + 18], title, '#f5cf79');
            frame.axes.forEach((p, k) => {
                const end = project(p);
                scene.arrow(origin, end, axisColors[k], 2, alpha);
                text([end[0] + 5, end[1] - 5], ['X', 'Y', 'Z'][k], axisColors[k]);
            });
        };
        drawFrame(bodyFrame(data.states[0], 1), 'Inicial', .35);
        drawFrame(bodyFrame(state, scale), 'Pasivo', 1);
        drawFrame(bodyFrame(state,scale,'active'),'Activo',.85);
        const origin = Array(data.dimension).fill(0);
        origin[1] = -.065;
        drawFrame({
            origin,
            axes: origin.map((_, k) => add(origin, origin.map((_, j) => j === k ? .014 : 0)))
        }, 'Global', .8);
    }
    if ($('trajectories').checked) {
        const c0 = center(data.positions);
        scene.polyline(data.target_path.map(s => project(add(c0, mul(s.output.slice(0, data.dimension), scale)))), '#eab4fa', 2);
    }
    drawMotionComparison(state,scale);
    scene.paint();
    $('labels').innerHTML = labels;
    if (window.populationRenderedFor !== data) {
    $('batchEfficiency').textContent=data.batch.present_beam_slots ? `${data.batch.evaluation_groups} lotes · ${number(100/data.batch.topology_padding_ratio)} % de las vigas calculadas se utilizan. El resto corresponde a conexiones ausentes en algunas alternativas.` : '';
    $('batchStatus').textContent = `${data.population.length} individuos · lote ${data.batch.batch_size} · ${data.batch.converged} convergentes · ${data.batch.rejected} rechazados · ${number(data.batch.seconds)} s`;
    $('population').innerHTML = data.population.map(p => {
        const points = p.positions.map(v => [Math.cos(.5)*v[0]-Math.sin(.5)*v[2],Math.cos(.3)*v[1]-Math.sin(.3)*(Math.sin(.5)*v[0]+Math.cos(.5)*v[2])]),
            xs = points.map(p => p[0]),
            ys = points.map(p => p[1]);
        const lo = [Math.min(...xs), Math.min(...ys)],
            hi = [Math.max(...xs), Math.max(...ys)];
        const factor = Math.min(90 / Math.max(hi[0] - lo[0], .01), 46 / Math.max(hi[1] - lo[1], .01));
        const xy = v => [55 + (v[0] - (lo[0] + hi[0]) / 2) * factor, 30 - (v[1] - (lo[1] + hi[1]) / 2) * factor];
        const lines = data.edges.map(([i, j], k) => {
            if (!p.presence[k]) return '';
            const a = xy(points[i]),
                b = xy(points[j]);
            return `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#317d73" stroke-width="1.3"/>`;
        }).join('');
        return `<button class="candidate ${p.index===data.selected_index?'selected':''}" data-index="${p.index}" aria-label="Ver alternativa ${p.index+1}" aria-pressed="${p.index===data.selected_index}"><span class="candidate-header"><span class="candidate-rank">${p.index===0?'Mejor':'#'+(p.index+1)}</span>${p.index===data.selected_index?'<span class="candidate-current">En visor</span>':''}</span><svg viewBox="0 0 110 60" aria-hidden="true">${lines}</svg><span class="candidate-quality"><span class="candidate-score"><span>Coste</span> ${number(p.objective)}</span><em>${p.index===0?'Referencia':'+'+number(p.gap_percent)+' %'}</em></span><small title="Objetivo normalizado y similitud media; usado para elegir padres">Selección ${number(p.selection_score)} · similitud ${Math.round(p.similarity*100)} %</small><small>${p.beams} vigas · ${p.topology_changes} conexiones distintas</small></button>`;
    }).join('');
    $('evolutionSummary').textContent = data.batch.focused_children !== undefined ? `${data.batch.focused_children} variantes de los ${data.batch.parent_pool} padres por calidad y diversidad · ${data.batch.exploratory_children} descendientes exploratorios · ${data.batch.offspring_gradient_steps} pasos de gradiente antes de seleccionar (${data.batch.offspring_improved}/${data.batch.offspring_refined} mejoran)` : 'La evolución concentra variantes cerca de los mejores y conserva una cuota de exploración.';
    $('population').querySelectorAll('button').forEach(button => button.onclick = () => action('/api/state?index=' + button.dataset.index));
    const steps=data.search_config.gradient_steps_per_evolution;
    $('cycleStatus').textContent=`Ciclo: ${steps} pasos de gradiente y una generación de evolución · siguiente: ${data.hybrid_phase<steps?`gradiente ${data.hybrid_phase+1}/${steps}`:'evolución'}`;
    $('stats').innerHTML = [['Iteración',data.iteration],['Coste',number(data.objective)],['Estados FEM',data.states.length],['Vigas',data.presence.filter(x=>x===1).length],['Dispositivo',data.device_name]].map(([label,value])=>`<div class="stat"><dt>${label}</dt><dd>${value}</dd></div>`).join('');
        window.populationRenderedFor=data;
    }
    const names = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz'];
    $('state').textContent = `Progreso: ${number(state.factor)} · entrada–salida\n` + state.output.map((v, i) => `${names[i]}: ${number(v*(i<data.dimension?1000:180/Math.PI))} ${i<data.dimension?'mm':'°'}`).join('\n') + `\nDiagnóstico del estado FEM en progreso ${number(state.equilibrium_factor)} (sin interpolar):\nEnergía: ${number(state.energy)} J\nResiduo: ${number(state.residual)}\nRigidez mínima: ${number(state.stiffness[0])} N/m\nRigidez siguiente: ${number(state.stiffness[1])} N/m\nNewton: ${state.iterations}`;
    const geometry=data.geometry_diagnostics;
    $('geometrySummary').textContent=geometry?`Recorrido · ${geometry.colliding_pairs} pares en contacto · penetración máxima ${number(geometry.maximum_penetration_mm)} mm · coste de colisiones ${number(geometry.collision_cost)}. Longitudes ${number(geometry.minimum_beam_length_mm)}–${number(geometry.maximum_beam_length_mm)} mm · coste de longitudes ${number(geometry.length_cost)}. ${geometry.samples} configuraciones comprobadas.`:'Sin recorrido convergente para comprobar la geometría.';
    const physical=data.physical_diagnostics;
    if(physical)$('state').textContent += `\nRecorrido completo · modelo de vigas:\nDeformación normal máxima: ${number(physical.maximum_normal_strain*100)} %\nDeformación cortante por torsión máxima: ${number(physical.maximum_torsional_shear_strain*100)} %\nViga crítica: ${physical.critical_beam+1} · progreso ${number(physical.critical_factor)}\nFuerza máxima del actuador: ${number(physical.maximum_actuator_force)} N\nMomento generalizado máximo: ${number(physical.maximum_actuator_generalized_moment)} N·m\nNo incluye concentraciones en uniones ni criterio de fallo del material.`;
    plot(state, names);
    drawTrajectoryOverview(state);
}

function plot(state, names) {
    const index = Number($('coordinate').value),
        multiplier = index < data.dimension ? 1000 : 180 / Math.PI,
        unit = index < data.dimension ? 'mm' : '°';
    const values = data.states.map(s => s.output[index]),
        target = data.target[index],
        targets = data.target_path.map(s => s.output[index]);
    const minimum = Math.min(0, ...values, ...targets),
        maximum = Math.max(0, ...values, ...targets),
        range = Math.max(maximum - minimum, 1e-8);
    const xy = (factor, v) => [50 + factor * 550, 170 - (v - minimum) / range * 130];
    const line = (a, b, color, width = 1) => `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}"/>`;
    const selected = xy(state.factor, state.output[index]);
    $('plot').innerHTML = `<text x="50" y="20" fill="#6b756f" font-size="12">${names[index]} (${unit}) · ${number(minimum*multiplier)} a ${number(maximum*multiplier)}</text>${line(xy(0,0),xy(1,0),'#cfd6d0')}<polyline points="${data.target_path.map(s=>xy(s.factor,s.output[index]).join(',')).join(' ')}" fill="none" stroke="#8b629b" stroke-width="2"/><polyline points="${data.states.map(s=>xy(s.factor,s.output[index]).join(',')).join(' ')}" fill="none" stroke="#23786c" stroke-width="2"/><circle cx="${selected[0]}" cy="${selected[1]}" r="4" fill="#23786c"/><text x="50" y="205" fill="#23786c" font-size="12">Real</text><text x="100" y="205" fill="#8b629b" font-size="12">Objetivo</text><text x="450" y="205" fill="#6b756f" font-size="12">Progreso del recorrido →</text>`;
}
$('gradient').onclick = () => action('/api/step', {
    method: 'gradient'
});
$('evolution').onclick = () => action('/api/step', {
    method: 'evolution'
});
$('run').onclick = () => {
    running = !running;
    $('run').textContent = running ? 'Detener búsqueda' : 'Iniciar búsqueda';
    if (running && !busy) action('/api/step', {
        method: $('method').value
    });
};
$('play').onclick = () => {
    playing = !playing;
    $('play').textContent = playing ? 'Pausar' : 'Animar';
};
let playbackTime=null;
function animatePlayback(time){
    if(playing && data?.states.length && !document.hidden && !$('synthesisScreen').hidden){
        if(playbackTime!==null && time-playbackTime<1000/30){requestAnimationFrame(animatePlayback);return;}
        if(playbackTime!==null){
            const maximum=Number($('frame').max);
            $('frame').value=(Number($('frame').value)+Math.min(100,time-playbackTime)/8)%(maximum+1);
            render();
        }
        playbackTime=time;
    }else playbackTime=null;
    requestAnimationFrame(animatePlayback);
}
requestAnimationFrame(animatePlayback);
for (const id of ['frame', 'frames', 'trajectories', 'coordinate', 'compare', 'targetBody']) $(id).oninput = render;
$('scale').oninput = () => {
    automaticScale = false;
    render();
};
$('autoScale').onclick = () => {
    automaticScale = true;
    fitDeformation();
    render();
};
let drag;
let orbitFrame = null, orbitTime = null;
function orbit(time) {
    orbitFrame = null;
    if (!viewerRotating()) {
        orbitTime = null;
        return;
    }
    const box=$('view').getBoundingClientRect();
    const visible = data && !document.hidden && !$('synthesisScreen').hidden && !drag && box.bottom>0 && box.top<innerHeight;
    if (!visible) orbitTime = null;
    else if (orbitTime === null) orbitTime = time;
    else if (time - orbitTime >= 1000 / 30) {
        const elapsed = Math.min(time - orbitTime, 100) / 1000;
        yaw = (yaw + elapsed * Number($('orbitSpeed').value) * Math.PI / 180) % (2 * Math.PI);
        orbitTime = time;
        render();
    }
    orbitFrame = requestAnimationFrame(orbit);
}
function viewerRotating(){return $('autoRotate').getAttribute('aria-pressed')==='true';}
$('autoRotate').onclick = () => {
    const enabled=!viewerRotating();
    $('autoRotate').setAttribute('aria-pressed',String(enabled));
    $('autoRotate').textContent=enabled?'Detener rotación':'Rotación automática';
    orbitTime=null;
    if(orbitFrame!==null)cancelAnimationFrame(orbitFrame);
    orbitFrame=enabled?requestAnimationFrame(orbit):null;
};
$('orbitSpeed').oninput = () => $('orbitSpeedValue').textContent = `${$('orbitSpeed').value}°/s`;
$('view').onpointerdown = e => {
    drag = [e.clientX, e.clientY];
    $('view').setPointerCapture(e.pointerId);
};
$('view').onpointermove = e => {
    if (drag && data?.dimension === 3) {
        yaw += (e.clientX - drag[0]) * .01;
        pitch += (e.clientY - drag[1]) * .01;
        drag = [e.clientX, e.clientY];
        render();
    }
};
$('view').onpointerup = () => drag = null;
$('view').onpointercancel = () => drag = null;
$('export').onclick = () => {
    if (!data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json'
    }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mechanism-path.json';
    link.click();
    URL.revokeObjectURL(url);
};
window.addEventListener('resize', render);
try {
    scene = new WebGLScene($('view'));
    action('/api/state');
} catch (e) {
    $('status').textContent = e.message;
}
