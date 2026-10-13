/* Passive center trajectories and physical orientations share one fitted scale.
 * Static paths rebuild only for a new snapshot, camera or frame density.
 */
let trajectorySource=null, trajectoryGeometry=null;
let trajectoryYaw=.62, trajectoryPitch=.42, trajectoryDrag=null;
const trajectoryColors={real:'#23786c',target:'#8b629b'};

function trajectoryProjection(point) {
    const view=$('trajectoryView').value;
    if(view==='xy')return [point[0],point[1]];
    if(view==='xz')return [point[0],point[2]];
    if(view==='yz')return [point[1],point[2]];
    const a=Math.cos(trajectoryYaw)*point[0]-Math.sin(trajectoryYaw)*point[2];
    const b=Math.sin(trajectoryYaw)*point[0]+Math.cos(trajectoryYaw)*point[2];
    return [a,Math.cos(trajectoryPitch)*point[1]-Math.sin(trajectoryPitch)*b];
}

function trajectoryOrigin(pose) { return pose.slice(0,data.dimension).map(v=>v*1000); }
function trajectoryAxes(pose,length) {
    const origin=trajectoryOrigin(pose);
    return Array.from({length:data.dimension},(_,k)=>add(origin,rotate(
        Array.from({length:data.dimension},(_,j)=>j===k?length:0),pose.slice(data.dimension)
    )));
}
function trajectoryFrame(pose,factor,kind,current=false) {
    const g=trajectoryGeometry, color=trajectoryColors[kind];
    const start=g.xy(trajectoryOrigin(pose));
    const axes=trajectoryAxes(pose,g.axisLength*(current?1.35:1));
    const lines=axes.map((axis,k)=>{
        const end=g.xy(axis),dx=end[0]-start[0],dy=end[1]-start[1],length=Math.hypot(dx,dy);
        if(length<.1)return '';
        const u=[dx/length,dy/length],back=[end[0]-u[0]*5,end[1]-u[1]*5];
        const arrow=`M ${back[0]-u[1]*2.5} ${back[1]+u[0]*2.5} L ${end[0]} ${end[1]} L ${back[0]+u[1]*2.5} ${back[1]-u[0]*2.5}`;
        const dashed=kind==='target'?'4 3':k===1?'2 2':k===2?'1 3':'';
        return `<line data-axis="${k}" x1="${start[0]}" y1="${start[1]}" x2="${end[0]}" y2="${end[1]}" stroke="${color}" stroke-width="${current?2.2:1.4}" stroke-dasharray="${dashed}"/><path d="${arrow}" fill="none" stroke="${color}" stroke-width="1.3"/>${current?`<text x="${end[0]+4}" y="${end[1]-4}" fill="${color}" font-size="11">${['X','Y','Z'][k]}</text>`:''}`;
    }).join('');
    return `<g class="trajectory-reference ${kind}" data-progress="${factor}" opacity="${current?1:.68}"><title>${kind==='real'?'Real':'Deseada'} · progreso ${(factor*100).toFixed(1)} %</title>${lines}<circle cx="${start[0]}" cy="${start[1]}" r="${current?4:2.5}" fill="${kind==='real'?color:'#fff'}" stroke="${color}" stroke-width="1.5"/>${!current&&kind==='target'?`<text x="${start[0]+7}" y="${start[1]+16}" fill="${color}" font-size="10">${Math.round(factor*100)} %</text>`:''}</g>`;
}
function prepareTrajectoryOverview() {
    const d=data.dimension,count=Number($('trajectoryFrameCount').value);
    const first=data.states[0].factor,last=data.states.at(-1).factor;
    const factors=[...new Set([...data.states.map(s=>s.factor),...Array.from({length:129},(_,i)=>first+(last-first)*i/128)])].sort((a,b)=>a-b);
    const actual=factors.map(factor=>({factor,output:passivePoseAt(factor)}));
    const desired=data.target_path;
    const frames=Array.from({length:count},(_,i)=>first+(last-first)*i/(count-1));
    const pairs=frames.map(factor=>({factor,real:passivePoseAt(factor),target:targetPose(factor)}));
    const centers=[...actual,...desired].map(s=>trajectoryOrigin(s.output));
    const span=Math.max(.2,...Array.from({length:d},(_,k)=>Math.max(...centers.map(p=>p[k]))-Math.min(...centers.map(p=>p[k]))));
    const axisLength=span*.07;
    const all=[Array(d).fill(0),...centers,...pairs.flatMap(pair=>[...trajectoryAxes(pair.real,axisLength*1.35),...trajectoryAxes(pair.target,axisLength*1.35)])].map(trajectoryProjection);
    const low=[0,1].map(k=>Math.min(...all.map(p=>p[k]))),high=[0,1].map(k=>Math.max(...all.map(p=>p[k])));
    const scale=Math.min(690/Math.max(high[0]-low[0],.2),310/Math.max(high[1]-low[1],.2));
    const xy=point=>{const p=trajectoryProjection(point);return [420+(p[0]-(low[0]+high[0])/2)*scale,190-(p[1]-(low[1]+high[1])/2)*scale];};
    trajectoryGeometry={xy,axisLength,first,last};
    let grid='';
    const origin=[58,370];
    grid+=Array.from({length:3},(_,k)=>{
        const basis=Array.from({length:3},(_,j)=>j===k?1:0),p=trajectoryProjection(basis);
        const end=[origin[0]+p[0]*25,origin[1]-p[1]*25];
        return `<line x1="${origin[0]}" y1="${origin[1]}" x2="${end[0]}" y2="${end[1]}" stroke="#778178"/><text x="${end[0]+4}" y="${end[1]-4}" fill="#6b756f" font-size="11">${['X','Y','Z'][k]}</text>`;
    }).join('');
    const path=(samples,kind)=>`<polyline class="trajectory-path ${kind}" points="${samples.map(s=>xy(trajectoryOrigin(s.output)).join(',')).join(' ')}" fill="none" stroke="${trajectoryColors[kind]}" stroke-width="2.4" ${kind==='target'?'stroke-dasharray="7 4"':''}/>`;
    $('trajectoryOverviewBase').innerHTML=grid+path(desired,'target')+path(actual,'real')+pairs.map(pair=>trajectoryFrame(pair.target,pair.factor,'target')+trajectoryFrame(pair.real,pair.factor,'real')).join('');
}
function drawTrajectoryOverview(state) {
    $('trajectoryOverview').classList.toggle('rotatable',$('trajectoryView').value==='orbit');
    const camera=[trajectoryYaw,trajectoryPitch,$('trajectoryView').value,$('trajectoryFrameCount').value].join(',');
    if(!state){
        trajectorySource=null;trajectoryGeometry=null;
        $('trajectoryOverviewBase').innerHTML='<text x="400" y="210" text-anchor="middle" fill="#6b756f" font-size="15">No hay recorrido real convergente para comparar.</text>';
        $('trajectoryOverviewCurrent').innerHTML='';$('trajectoryProgress').textContent='Sin recorrido';return;
    }
    if(trajectorySource!==data || trajectoryGeometry?.camera!==camera){
        trajectorySource=data;prepareTrajectoryOverview();trajectoryGeometry.camera=camera;
    }
    const desired=targetPose(state.factor),a=trajectoryGeometry.xy(trajectoryOrigin(state.output)),b=trajectoryGeometry.xy(trajectoryOrigin(desired));
    $('trajectoryOverviewCurrent').innerHTML=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#a5b1a6" stroke-dasharray="3 4"/>`+trajectoryFrame(desired,state.factor,'target',true)+trajectoryFrame(state.output,state.factor,'real',true);
    $('trajectoryProgress').textContent=`Progreso ${(state.factor*100).toFixed(1)} %`;
    $('trajectoryOverviewCaption').textContent=`Centro del pasivo respecto al reposo, en mm. Marcos al mismo progreso; pulsa uno para comparar ese instante. En «3D libre», arrastra para girar.${trajectoryGeometry.last<.999999?` El recorrido real alcanza ${(trajectoryGeometry.last*100).toFixed(1)} %.`:''}`;
}
// Handlers use the application's snapshot and renderer once they are initialized.
for(const id of ['trajectoryFrameCount','trajectoryView'])document.getElementById(id).addEventListener('change',()=>render());
const trajectorySvg=document.getElementById('trajectoryOverview');
trajectorySvg.addEventListener('pointerdown',e=>{
    if($('trajectoryView').value!=='orbit' || e.button!==0)return;
    trajectoryDrag={x:e.clientX,y:e.clientY,moved:false,factor:e.target.closest('.trajectory-reference')?.dataset.progress};trajectorySvg.setPointerCapture(e.pointerId);
});
trajectorySvg.addEventListener('pointermove',e=>{
    if(!trajectoryDrag)return;
    const dx=e.clientX-trajectoryDrag.x,dy=e.clientY-trajectoryDrag.y;
    if(Math.abs(dx)+Math.abs(dy)>1)trajectoryDrag.moved=true;
    trajectoryYaw+=dx*.008;trajectoryPitch=Math.max(-1.5,Math.min(1.5,trajectoryPitch+dy*.008));
    trajectoryDrag.x=e.clientX;trajectoryDrag.y=e.clientY;render();
});
trajectorySvg.addEventListener('pointerup',e=>{
    const moved=trajectoryDrag?.moved,factor=trajectoryDrag?.factor ?? e.target.closest('.trajectory-reference')?.dataset.progress;trajectoryDrag=null;
    if(trajectorySvg.hasPointerCapture(e.pointerId))trajectorySvg.releasePointerCapture(e.pointerId);
    if(moved)return;
    if(factor!==undefined){$('frame').value=Math.round(Number(factor)*1000);render();}
});
trajectorySvg.addEventListener('pointercancel',()=>{trajectoryDrag=null;});

let trajectoryOrbitFrame=null,trajectoryOrbitTime=null;
function trajectoryOrbit(time){
    if($('trajectoryAutoRotate').getAttribute('aria-pressed')!=='true'){trajectoryOrbitFrame=null;return;}
    const box=trajectorySvg.getBoundingClientRect();
    const visible=data?.states.length && !document.hidden && !$('synthesisScreen').hidden && box.bottom>0 && box.top<innerHeight && !trajectoryDrag;
    if(!visible)trajectoryOrbitTime=null;
    else if(trajectoryOrbitTime===null)trajectoryOrbitTime=time;
    else if(time-trajectoryOrbitTime>=1000/30){
        trajectoryYaw+=Math.min(100,time-trajectoryOrbitTime)*Math.PI/9000;
        trajectoryOrbitTime=time;
        const factor=Math.min(data.states.at(-1).factor,Math.max(data.states[0].factor,Number($('frame').value)/1000));
        drawTrajectoryOverview({factor,output:passivePoseAt(factor)});
    }
    trajectoryOrbitFrame=requestAnimationFrame(trajectoryOrbit);
}
document.getElementById('trajectoryAutoRotate').onclick=()=>{
    const enabled=$('trajectoryAutoRotate').getAttribute('aria-pressed')!=='true';
    $('trajectoryAutoRotate').setAttribute('aria-pressed',String(enabled));
    $('trajectoryAutoRotate').textContent=enabled?'Detener rotación':'Rotación automática';
    if(enabled)$('trajectoryView').value='orbit';
    trajectoryOrbitTime=null;
    if(trajectoryOrbitFrame!==null)cancelAnimationFrame(trajectoryOrbitFrame);
    trajectoryOrbitFrame=enabled?requestAnimationFrame(trajectoryOrbit):null;
    render();
};
