/* Interpolation is visual only; computed FEM states remain unchanged. */
function quaternion(r) {
    const angle=Math.hypot(...r), s=angle<1e-12?.5:Math.sin(angle/2)/angle;
    return [Math.cos(angle/2),...r.map(x=>x*s)];
}
function interpolateRotation(a,b,t) {
    const qa=quaternion(a);let qb=quaternion(b);
    let dot=qa.reduce((s,x,i)=>s+x*qb[i],0);
    if(dot<0){qb=qb.map(x=>-x);dot=-dot;}
    dot=Math.min(1,dot);
    const angle=Math.acos(dot), sine=Math.sin(angle);
    const q=qa.map((x,i)=>sine<1e-8?(1-t)*x+t*qb[i]:(Math.sin((1-t)*angle)*x+Math.sin(t*angle)*qb[i])/sine);
    const norm=Math.hypot(...q);for(let i=0;i<4;i++)q[i]/=norm;
    const v=Math.hypot(...q.slice(1)), scale=v<1e-12?2:2*Math.atan2(v,q[0])/v;
    return q.slice(1).map(x=>x*scale);
}
function passivePoseAt(progress) {
    const states=data.states;
    let i=0;while(i<states.length-2 && states[i+1].factor<progress)i++;
    const a=states[i],b=states[Math.min(i+1,states.length-1)];
    const t=Math.min(1,Math.max(0,(progress-a.factor)/Math.max(b.factor-a.factor,1e-12)));
    return [...a.output.slice(0,data.dimension).map((v,k)=>v+t*(b.output[k]-v)),...interpolateRotation(a.output.slice(data.dimension),b.output.slice(data.dimension),t)];
}
function animatedState(progress) {
    const states=data.states;
    let i=0;while(i<states.length-2 && states[i+1].factor<progress)i++;
    const a=states[i], b=states[Math.min(i+1,states.length-1)];
    const t=Math.min(1,Math.max(0,(progress-a.factor)/Math.max(b.factor-a.factor,1e-12)));
    const lerp=(x,y)=>x.map((v,k)=>v+(y[k]-v)*t);
    const output=passivePoseAt(progress);
    const factor=a.factor+t*(b.factor-a.factor), input=targetPose(factor,data.input_waypoints);
    const bodies={};for(const role of ['active','passive']){const pose=role==='active'?input:output,c0=center(data.positions,role);bodies[role]={pose,c0,c:add(c0,pose.slice(0,data.dimension))};}
    const beam_nodes=a.beam_nodes?a.beam_nodes.map((nodes,e)=>nodes.map((node,k)=>lerp(node,b.beam_nodes[e][k]))):null;
    const state = {...a,beam_nodes,equilibrium_factor:a.factor,factor,output,input,
        positions:a.positions.map((p,k)=>bodies[data.roles[k]]?add(bodies[data.roles[k]].c,rotate(sub(data.positions[k],bodies[data.roles[k]].c0),bodies[data.roles[k]].pose.slice(data.dimension))):lerp(p,b.positions[k])),
        rotations:a.rotations.map((r,k)=>bodies[data.roles[k]]?bodies[data.roles[k]].pose.slice(data.dimension):interpolateRotation(r,b.rotations[k],t))};
    if(beam_nodes)data.edges.forEach(([i,j],e)=>{beam_nodes[e][0]=[...state.positions[i],...state.rotations[i]];beam_nodes[e][beam_nodes[e].length-1]=[...state.positions[j],...state.rotations[j]];});
    return state;
}
function targetPose(t,waypoints=data.waypoints) {
    const f=waypoints.factors,p=waypoints.poses;
    let i=0;while(i<f.length-2 && t>=f[i+1])i++;
    const h=f[i+1]-f[i],u=Math.min(1,Math.max(0,(t-f[i])/h));
    const slopes=p.slice(1).map((q,j)=>q.map((v,k)=>(v-p[j][k])/(f[j+1]-f[j])));
    const tangent=j=>j===0?slopes[0]:j===p.length-1?slopes.at(-1):slopes[j].map((v,k)=>(v+slopes[j-1][k])/2);
    return p[i].map((v,k)=>(2*u**3-3*u**2+1)*v+(u**3-2*u**2+u)*h*tangent(i)[k]+(-2*u**3+3*u**2)*p[i+1][k]+(u**3-u**2)*h*tangent(i+1)[k]);
}
function drawMotionComparison(state,scale) {
    if(!$('targetBody').checked){$('poseComparison').textContent='';$('orientationCompare').innerHTML='';return;}
    const desired=targetPose(state.factor),c0=center(data.positions),d=data.dimension;
    const origin=add(c0,mul(desired.slice(0,d),scale)),real=bodyFrame(state,scale);
    const ids=mobileIds(), points=ids.map(i=>add(origin,rotate(sub(data.positions[i],c0),desired.slice(d))));
    for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){
        const a=project(points[i]),b=project(points[j]),delta=sub(b,a),length=Math.hypot(...delta);
        for(let s=0;s<length;s+=12)scene.line(add(a,mul(delta,s/length)),add(a,mul(delta,Math.min(s+7,length)/length)),'#eab4fa',3,.85);
    }
    points.forEach(p=>scene.circle(project(p),5,'#eab4fa'));
    scene.line(project(origin),project(real.origin),'#eab4fa',1,.7);
    scene.circle(project(origin),4,'#eab4fa');scene.circle(project(real.origin),4,'#6fe0ba');
    if($('frames').checked)for(let k=0;k<d;k++){
        const v=Array.from({length:d},(_,j)=>j===k?.014:0);
        scene.arrow(project(origin),project(add(origin,rotate(v,desired.slice(d)))),'#eab4fa',1.5,.85);
    }
    const position=Math.hypot(...sub(state.output.slice(0,d),desired.slice(0,d)))*1000;
    const angle=2*Math.acos(Math.min(1,Math.abs(quaternion(state.output.slice(d)).reduce((s,x,i)=>s+x*quaternion(desired.slice(d))[i],0))));
    $('poseComparison').textContent=`Progreso ${(state.factor*100).toFixed(1)} % · diferencia ${position.toFixed(3)} mm · ${(angle*180/Math.PI).toFixed(2)}°`;
    // Shared origin removes translation, making orientation directly comparable.
    const zero=Array(d).fill(0),p0=project(zero);
    const xy=v=>{const p=project(v);return [100+(p[0]-p0[0])*2,85+(p[1]-p0[1])*2];};
    const axes=(r,color)=>Array.from({length:d},(_,k)=>{
        const v=Array.from({length:d},(_,j)=>j===k?.006:0), p=xy(rotate(v,r));
        return `<line x1="100" y1="85" x2="${p[0]}" y2="${p[1]}" stroke="${color}" stroke-width="2"/><circle cx="${p[0]}" cy="${p[1]}" r="3" fill="${color}"/><text x="${p[0]+4}" y="${p[1]-4}" fill="${color}" font-size="10">${['X','Y','Z'][k]}</text>`;
    }).join('');
    $('orientationCompare').innerHTML=`<text x="12" y="18" fill="#b6ccd8" font-size="11">Orientación · mismo origen</text>${axes(desired.slice(d),'#eab4fa')}${axes(state.output.slice(d),'#6fe0ba')}`;
}
