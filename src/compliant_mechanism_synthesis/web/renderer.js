/* WebGL2 scene geometry is rasterized on the browser's graphics device. */
class WebGLScene {
    constructor(canvas) {
        this.canvas = canvas;
        const gl = this.gl = canvas.getContext('webgl2', {
            antialias: true,
            alpha: true
        });
        if (!gl) throw Error('El visor necesita WebGL2 habilitado en el navegador.');
        const shader = (kind, source) => {
            const item = gl.createShader(kind);
            gl.shaderSource(item, source);
            gl.compileShader(item);
            if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(item));
            return item;
        };
        this.program = gl.createProgram();
        gl.attachShader(this.program, shader(gl.VERTEX_SHADER, `#version 300 es
      in vec2 position; in vec4 color; out vec4 tint;
      uniform vec2 resolution; uniform float sceneScale;
      void main(){ vec2 p=(position-vec2(450.,300.))*sceneScale+resolution/2.;
        gl_Position=vec4(p/resolution*vec2(2.,-2.)+vec2(-1.,1.),0.,1.); tint=color; }`));
        gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float; in vec4 tint; out vec4 outputColor;
      void main(){outputColor=tint;}`));
        gl.linkProgram(this.program);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(this.program));
        this.buffer = gl.createBuffer();
        this.upload = new Float32Array(0);
        canvas.addEventListener('webglcontextlost', e => {
            e.preventDefault();
            document.getElementById('status').textContent = 'Se perdió el contexto WebGL; recarga el visor.';
        });
    }
    begin() {
        this.vertices = [];
    }
    color(hex, alpha = 1) {
        return [1, 3, 5].map(k => parseInt(hex.slice(k, k + 2), 16) / 255).concat(alpha);
    }
    triangle(a, b, c, color) {
        for (const p of [a, b, c]) this.vertices.push(...p, ...color);
    }
    line(a, b, hex, width = 1, alpha = 1) {
        const dx = b[0] - a[0],
            dy = b[1] - a[1],
            length = Math.hypot(dx, dy);
        if (length < 1e-8) return;
        const nx = -dy / length * width / 2,
            ny = dx / length * width / 2;
        const p = [a[0] + nx, a[1] + ny],
            q = [a[0] - nx, a[1] - ny],
            r = [b[0] + nx, b[1] + ny],
            s = [b[0] - nx, b[1] - ny];
        const color = this.color(hex, alpha);
        this.triangle(p, q, r, color);
        this.triangle(q, s, r, color);
    }
    polyline(points, color, width = 1, alpha = 1) {
        for (let i = 1; i < points.length; i++) this.line(points[i - 1], points[i], color, width, alpha);
    }
    circle(p, r, hex, alpha = 1) {
        const color = this.color(hex, alpha);
        for (let k = 0; k < 24; k++) this.triangle(p, [p[0] + r * Math.cos(k * Math.PI / 12), p[1] + r * Math.sin(k * Math.PI / 12)], [p[0] + r * Math.cos((k + 1) * Math.PI / 12), p[1] + r * Math.sin((k + 1) * Math.PI / 12)], color);
    }
    square(p, r, hex) {
        const c = this.color(hex);
        this.triangle([p[0] - r, p[1] - r], [p[0] + r, p[1] - r], [p[0] + r, p[1] + r], c);
        this.triangle([p[0] - r, p[1] - r], [p[0] + r, p[1] + r], [p[0] - r, p[1] + r], c);
    }
    arrow(a, b, color, width = 2, alpha = 1) {
        this.line(a, b, color, width, alpha);
        const dx = b[0] - a[0],
            dy = b[1] - a[1],
            length = Math.hypot(dx, dy);
        if (length < 1) return;
        const x = dx / length,
            y = dy / length;
        this.triangle(b, [b[0] - x * 8 - y * 4, b[1] - y * 8 + x * 4], [b[0] - x * 8 + y * 4, b[1] - y * 8 - x * 4], this.color(color, alpha));
    }
    paint() {
        const gl = this.gl,
            c = this.canvas,
            ratio = window.devicePixelRatio || 1;
        const width = Math.round(c.clientWidth * ratio),
            height = Math.round(c.clientHeight * ratio);
        if (c.width !== width || c.height !== height) {
            c.width = width;
            c.height = height;
        }
        gl.viewport(0, 0, c.width, c.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        if (this.upload.length < this.vertices.length) {
            this.upload = new Float32Array(Math.max(1024, this.vertices.length * 2));
            gl.bufferData(gl.ARRAY_BUFFER, this.upload.byteLength, gl.DYNAMIC_DRAW);
        }
        if (this.vertices.length) {
            this.upload.set(this.vertices);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.upload, 0, this.vertices.length);
        }
        for (const [name, count, offset] of [
                ['position', 2, 0],
                ['color', 4, 8]
            ]) {
            const location = gl.getAttribLocation(this.program, name);
            gl.enableVertexAttribArray(location);
            gl.vertexAttribPointer(location, count, gl.FLOAT, false, 24, offset);
        }
        gl.uniform2f(gl.getUniformLocation(this.program, 'resolution'), c.width, c.height);
        gl.uniform1f(gl.getUniformLocation(this.program, 'sceneScale'), Math.min(c.width / 900, c.height / 600));
        gl.drawArrays(gl.TRIANGLES, 0, this.vertices.length / 6);
    }
}
window.WebGLScene = WebGLScene;
