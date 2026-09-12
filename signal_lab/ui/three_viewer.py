"""Local Three.js WebGL viewer for Project SIGNAL particle states."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Callable
from urllib.parse import urlparse


STATE_PROVIDER = Callable[[], dict[str, object]]


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project SIGNAL - Three.js Particle View</title>
  <style>
    :root { color-scheme: dark; --ink: #e9f2f8; --muted: #91a8b7; --line: #294354; --panel: rgba(8, 21, 31, .82); }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #02070b; font-family: "Segoe UI", system-ui, sans-serif; color: var(--ink); }
    #stage { position: fixed; inset: 0; }
    canvas { display: block; width: 100%; height: 100%; }
    .hud { position: fixed; inset: 18px 20px auto 20px; display: flex; align-items: flex-start; justify-content: space-between; pointer-events: none; }
    .brand, .stats, .legend, .hint { background: var(--panel); border: 1px solid rgba(64, 105, 127, .72); box-shadow: 0 12px 35px rgba(0,0,0,.26); backdrop-filter: blur(10px); }
    .brand { padding: 12px 15px; letter-spacing: .02em; }
    .brand strong { display: block; font-size: 16px; }
    .brand span { display: block; color: var(--muted); font-size: 11px; margin-top: 4px; }
    .stats { min-width: 190px; padding: 10px 13px; font-size: 12px; line-height: 1.65; }
    .stats b { color: #ffffff; }
    .bottom { position: fixed; inset: auto 20px 18px 20px; display: flex; align-items: flex-end; justify-content: space-between; pointer-events: none; }
    .legend { display: flex; gap: 14px; padding: 9px 12px; font-size: 12px; }
    .legend i { display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; box-shadow: 0 0 10px currentColor; }
    .hint { padding: 8px 11px; color: var(--muted); font-size: 11px; }
    .detail { min-width: 250px; max-width: 360px; padding: 10px 13px; font-size: 11px; line-height: 1.55; }
    .detail b { color: #ffffff; }
    .debug-toggle { margin-left: 10px; padding: 6px 9px; border: 1px solid #3e6e83; border-radius: 4px; background: rgba(7, 19, 28, .88); color: #bfe8ff; cursor: pointer; font: 11px Consolas, monospace; }
    #offline { display: none; position: fixed; inset: 50% auto auto 50%; transform: translate(-50%, -50%); padding: 18px 22px; background: rgba(60, 21, 30, .94); border: 1px solid #d46a7a; color: white; }
  </style>
</head>
<body>
  <div id="stage"></div>
  <div class="hud">
    <div class="brand"><strong>Project SIGNAL</strong><span>Three.js WebGL particle field</span></div>
    <div class="stats" id="stats">Connecting...</div><button class="debug-toggle" id="networkDebug">NETWORK DEBUG: ON</button>
  </div>
  <div class="bottom">
    <div class="legend">
      <span style="color:#3b9cff"><i style="background:#3b9cff"></i>Species 0</span>
      <span style="color:#ffad3d"><i style="background:#ffad3d"></i>Species 1</span>
      <span style="color:#4fd27d"><i style="background:#4fd27d"></i>Species 2</span>
    </div>
    <div class="detail" id="detail">Click a cluster ring to inspect its links.</div>
    <div class="hint">Drag to orbit | Wheel to zoom | F11 for browser fullscreen</div>
  </div>
  <div id="offline">Waiting for the simulator connection...</div>
  <script type="module">
    import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js";

    const palette = [0x3b9cff, 0xffad3d, 0x4fd27d, 0xf05d5e, 0xb084cc, 0x42c6c9, 0xe77bdb, 0xd9e36a];
    const stage = document.getElementById("stage");
    const stats = document.getElementById("stats");
    const detail = document.getElementById("detail");
    const offline = document.getElementById("offline");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x02070b);
    const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 1, 10000);
    camera.position.set(0, 0, 1220);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    stage.appendChild(renderer.domElement);

    const world = new THREE.Group();
    scene.add(world);
    const particleGeometry = new THREE.BufferGeometry();
    const particleMaterial = new THREE.ShaderMaterial({
      uniforms: { uPixelRatio: { value: Math.min(devicePixelRatio, 2) } },
      vertexShader: `
        uniform float uPixelRatio;
        attribute float aSize;
        attribute float aSpecies;
        varying float vSpecies;
        void main() {
          vSpecies = aSpecies;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * uPixelRatio * (760.0 / max(160.0, -mvPosition.z));
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying float vSpecies;
        vec3 speciesColor(float value) {
          if (value < 0.5) return vec3(0.231, 0.612, 1.0);
          if (value < 1.5) return vec3(1.0, 0.678, 0.239);
          if (value < 2.5) return vec3(0.310, 0.824, 0.490);
          if (value < 3.5) return vec3(0.941, 0.365, 0.369);
          return vec3(0.690, 0.518, 0.800);
        }
        void main() {
          vec2 point = gl_PointCoord - vec2(0.5);
          float distanceFromCenter = length(point);
          if (distanceFromCenter > 0.5) discard;
          float glow = pow(1.0 - smoothstep(0.05, 0.5, distanceFromCenter), 1.7);
          vec3 color = speciesColor(vSpecies);
          gl_FragColor = vec4(color * (0.70 + glow * 1.55), glow * 0.96);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    world.add(particles);

    const haloGeometry = new THREE.BufferGeometry();
    const haloMaterial = new THREE.PointsMaterial({ color: 0x58b8ff, size: 24, transparent: true, opacity: 0.075, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
    const halos = new THREE.Points(haloGeometry, haloMaterial);
    world.add(halos);
    const linkLayer = new THREE.Group();
    world.add(linkLayer);
    const networkLayer = new THREE.Group();
    world.add(networkLayer);

    const grid = new THREE.Group();
    world.add(grid);
    let clusterRings = [];
    let clusterHitTargets = [];
    let networkHitTargets = [];
    let worldWidth = 1000;
    let worldHeight = 1000;
    let targetRotation = 0;
    let rotation = 0;
    let zoom = 1220;
    let gridWidth = 0;
    let gridHeight = 0;
    let pointerDown = false;
    let pointerX = 0;
    let pointerDownX = 0;
    let pointerDownY = 0;
    let activeClusterId = null;
    let currentPayload = null;
    let networkDebug = true;

    function makeGrid(width, height) {
      grid.clear();
      const material = new THREE.LineBasicMaterial({ color: 0x12303d, transparent: true, opacity: 0.42 });
      const points = [];
      for (let x = -width / 2; x <= width / 2; x += 100) points.push(x, -height / 2, -18, x, height / 2, -18);
      for (let y = -height / 2; y <= height / 2; y += 100) points.push(-width / 2, y, -18, width / 2, y, -18);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
      grid.add(new THREE.LineSegments(geometry, material));
      const borderPoints = [-width/2, -height/2, -20, width/2, -height/2, -20, width/2, height/2, -20, -width/2, height/2, -20, -width/2, -height/2, -20];
      const borderGeometry = new THREE.BufferGeometry();
      borderGeometry.setAttribute("position", new THREE.Float32BufferAttribute(borderPoints, 3));
      grid.add(new THREE.Line(borderGeometry, new THREE.LineBasicMaterial({ color: 0x315e71, transparent: true, opacity: 0.75 })));
    }

    function updateParticles(payload) {
      const positions = payload.positions || [];
      const velocities = payload.velocities || [];
      const species = payload.species || [];
      const count = species.length;
      const positionValues = new Float32Array(count * 3);
      const haloValues = new Float32Array(count * 3);
      const sizes = new Float32Array(count);
      const speciesValues = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        const x = (positions[i][0] || 0) - worldWidth / 2;
        const y = worldHeight / 2 - (positions[i][1] || 0);
        const vx = velocities[i]?.[0] || 0;
        const vy = velocities[i]?.[1] || 0;
        const speed = Math.min(1, Math.hypot(vx, vy) * 9);
        positionValues[i * 3] = x;
        positionValues[i * 3 + 1] = y;
        positionValues[i * 3 + 2] = speed * 22;
        haloValues[i * 3] = x;
        haloValues[i * 3 + 1] = y;
        haloValues[i * 3 + 2] = speed * 22 - 2;
        sizes[i] = 10 + speed * 6;
        speciesValues[i] = species[i];
      }
      particleGeometry.setAttribute("position", new THREE.BufferAttribute(positionValues, 3));
      particleGeometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
      particleGeometry.setAttribute("aSpecies", new THREE.BufferAttribute(speciesValues, 1));
      haloGeometry.setAttribute("position", new THREE.BufferAttribute(haloValues, 3));
      worldWidth = payload.width || worldWidth;
      worldHeight = payload.height || worldHeight;
      if (gridWidth !== worldWidth || gridHeight !== worldHeight) {
        gridWidth = worldWidth;
        gridHeight = worldHeight;
        makeGrid(worldWidth, worldHeight);
      }
    }

    function updateClusters(clusters) {
      for (const ring of clusterRings) world.remove(ring);
      for (const target of clusterHitTargets) world.remove(target);
      clusterRings = [];
      clusterHitTargets = [];
      for (const cluster of clusters || []) {
        const radius = Math.max(12, Number(cluster.radius || 12));
        const geometry = new THREE.RingGeometry(radius - 1.2, radius, 48);
        const selected = cluster.cluster_id === activeClusterId;
        const color = selected ? 0xffffff : (cluster.classification === "PERSISTENT" || cluster.classification === "LONG_LIVED" ? 0x4fd27d : 0x2b6078);
        const ring = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: selected ? 0.90 : 0.44, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
        ring.userData.clusterId = cluster.cluster_id;
        ring.position.set(Number(cluster.centroid_x || 0) - worldWidth / 2, worldHeight / 2 - Number(cluster.centroid_y || 0), 16);
        world.add(ring);
        clusterRings.push(ring);
        const hitTarget = new THREE.Mesh(new THREE.CircleGeometry(radius * 1.35, 48), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.001, depthWrite: false }));
        hitTarget.userData.clusterId = cluster.cluster_id;
        hitTarget.position.copy(ring.position);
        hitTarget.position.z = 15;
        world.add(hitTarget);
        clusterHitTargets.push(hitTarget);
      }
    }

    function labelSprite(text, color) {
      const canvas = document.createElement("canvas");
      canvas.width = 160;
      canvas.height = 44;
      const context = canvas.getContext("2d");
      context.font = "bold 24px Consolas";
      context.fillStyle = color;
      context.fillText(text, 6, 30);
      const texture = new THREE.CanvasTexture(canvas);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
      sprite.scale.set(68, 19, 1);
      return sprite;
    }

    function updateNetworks(networks) {
      networkLayer.clear();
      for (const target of networkHitTargets) world.remove(target);
      networkHitTargets = [];
      if (!networkDebug) return;
      for (const network of networks || []) {
        const nodes = network.nodes || [];
        const nodeById = new Map(nodes.map(node => [Number(node.node_id), node]));
        const edgeValues = [];
        for (const edge of network.edges || []) {
          const first = nodeById.get(Number(edge.node_a));
          const second = nodeById.get(Number(edge.node_b));
          if (!first || !second) continue;
          const x1 = Number(first.centroid?.[0] || 0) - worldWidth / 2;
          const y1 = worldHeight / 2 - Number(first.centroid?.[1] || 0);
          const x2 = Number(second.centroid?.[0] || 0) - worldWidth / 2;
          const y2 = worldHeight / 2 - Number(second.centroid?.[1] || 0);
          edgeValues.push(x1, y1, 34, x2, y2, 34);
        }
        if (edgeValues.length) {
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute("position", new THREE.Float32BufferAttribute(edgeValues, 3));
          networkLayer.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xf8d477, transparent: true, opacity: 0.82, linewidth: 3, blending: THREE.AdditiveBlending, depthWrite: false })));
        }
        for (const node of nodes) {
          const x = Number(node.centroid?.[0] || 0) - worldWidth / 2;
          const y = worldHeight / 2 - Number(node.centroid?.[1] || 0);
          const radius = Math.max(12, Number(node.radius || 12) + 8);
          const region = new THREE.Mesh(new THREE.CircleGeometry(radius, 40), new THREE.MeshBasicMaterial({ color: 0xf8d477, transparent: true, opacity: 0.055, side: THREE.DoubleSide, depthWrite: false }));
          region.position.set(x, y, 20);
          networkLayer.add(region);
          const marker = new THREE.Mesh(new THREE.SphereGeometry(4.5, 16, 10), new THREE.MeshBasicMaterial({ color: 0xfff0a8, transparent: true, opacity: 0.96, blending: THREE.AdditiveBlending, depthWrite: false }));
          marker.position.set(x, y, 38);
          networkLayer.add(marker);
          const label = labelSprite(`N${network.network_id}.${node.node_id}`, "#ffe69a");
          label.position.set(x + radius * 0.65, y - radius * 0.65, 42);
          networkLayer.add(label);
          const hitTarget = new THREE.Mesh(new THREE.CircleGeometry(Math.max(16, radius * 0.8), 32), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.001, depthWrite: false }));
          hitTarget.position.set(x, y, 39);
          hitTarget.userData.networkId = Number(network.network_id);
          hitTarget.userData.nodeId = Number(node.node_id);
          world.add(hitTarget);
          networkHitTargets.push(hitTarget);
        }
      }
    }

    function clusterColor(cluster) {
      return cluster.classification === "PERSISTENT" || cluster.classification === "LONG_LIVED" ? 0x4fd27d : 0x65b7d3;
    }

    function inspectCluster(clusterId) {
      if (!currentPayload) return;
      const cluster = (currentPayload.clusters || []).find(item => item.cluster_id === clusterId);
      if (!cluster) return;
      activeClusterId = clusterId;
      for (const ring of clusterRings) {
        const selected = ring.userData.clusterId === activeClusterId;
        ring.material.color.setHex(selected ? 0xffffff : clusterColor((currentPayload.clusters || []).find(item => item.cluster_id === ring.userData.clusterId) || {}));
        ring.material.opacity = selected ? 0.90 : 0.44;
      }
      const idToIndex = new Map((currentPayload.ids || []).map((id, index) => [id, index]));
      const members = (cluster.particle_ids || []).map(id => idToIndex.get(id)).filter(index => index !== undefined);
      const radius = Number(currentPayload.interaction_radius || 0);
      const size = [Number(currentPayload.width || worldWidth), Number(currentPayload.height || worldHeight)];
      const linkValues = [];
      let links = 0;
      for (let i = 0; i < members.length; i++) {
        const first = currentPayload.positions[members[i]];
        for (let j = i + 1; j < members.length; j++) {
          const second = currentPayload.positions[members[j]];
          const dx = (second[0] - first[0] + size[0] / 2) % size[0] - size[0] / 2;
          const dy = (second[1] - first[1] + size[1] / 2) % size[1] - size[1] / 2;
          if (Math.hypot(dx, dy) <= radius) {
            const x1 = first[0] - size[0] / 2;
            const y1 = size[1] / 2 - first[1];
            linkValues.push(x1, y1, 26, x1 + dx, y1 - dy, 26);
            links++;
          }
        }
      }
      linkLayer.clear();
      if (linkValues.length) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(linkValues, 3));
        linkLayer.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x9bdcff, transparent: true, opacity: 0.58, blending: THREE.AdditiveBlending, depthWrite: false })));
      }
      detail.innerHTML = `<b>C${cluster.cluster_id} - ${cluster.classification} (${cluster.size_class || ""})</b><br>Particles ${Number(cluster.particle_count || 0).toLocaleString()} | Age ${Number(cluster.age_steps || 0).toLocaleString()}<br>Structure Score ${Number(cluster.structure_score || 0).toFixed(2)} | Tracker Persistence ${Number(cluster.tracker_persistence_score || 0).toFixed(2)}<br>Cohesion ${Number(cluster.cohesion_score || 0).toFixed(2)} | Identity Score ${Number(cluster.identity_score || 0).toFixed(2)}<br>Shape ${Number(cluster.shape_score || 0).toFixed(2)} | Dynamics ${Number(cluster.dynamic_score || 0).toFixed(2)} | Bridge ${Number(cluster.bridge_fraction || 0).toFixed(2)}<br><b>${links.toLocaleString()} detected neighbor links</b>`;
      updateClusters(currentPayload.clusters);
    }

    function update(payload) {
      currentPayload = payload;
      if (activeClusterId !== null && !(payload.clusters || []).some(cluster => cluster.cluster_id === activeClusterId)) {
        activeClusterId = null;
        linkLayer.clear();
        detail.textContent = "Click a cluster ring to inspect its links.";
      }
      updateParticles(payload);
      updateClusters(payload.clusters);
      updateNetworks(payload.networks);
      stats.innerHTML = `<b>Step</b> ${Number(payload.step || 0).toLocaleString()}<br><b>Particles</b> ${Number(payload.particle_count || 0).toLocaleString()}<br><b>Structures</b> ${Number((payload.clusters || []).length)}<br><b>Interaction radius</b> ${Number(payload.interaction_radius || 0).toFixed(1)}`;
      offline.style.display = "none";
    }

    async function poll() {
      try {
        const response = await fetch(`/state?ts=${Date.now()}`, { cache: "no-store" });
        if (!response.ok) throw new Error("state unavailable");
        update(await response.json());
      } catch (error) {
        offline.style.display = "block";
      }
      setTimeout(poll, 70);
    }
    poll();

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    document.getElementById("networkDebug").addEventListener("click", () => {
      networkDebug = !networkDebug;
      document.getElementById("networkDebug").textContent = `NETWORK DEBUG: ${networkDebug ? "ON" : "OFF"}`;
      updateNetworks(currentPayload?.networks || []);
    });
    renderer.domElement.addEventListener("pointerdown", event => { pointerDown = true; pointerX = event.clientX; pointerDownX = event.clientX; pointerDownY = event.clientY; renderer.domElement.setPointerCapture(event.pointerId); });
    renderer.domElement.addEventListener("pointermove", event => { if (!pointerDown) return; targetRotation += (event.clientX - pointerX) * 0.004; pointerX = event.clientX; });
    renderer.domElement.addEventListener("pointerup", event => {
      const moved = Math.hypot(event.clientX - pointerDownX, event.clientY - pointerDownY);
      pointerDown = false;
      if (moved > 8) return;
      pointer.x = (event.clientX / innerWidth) * 2 - 1;
      pointer.y = -(event.clientY / innerHeight) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const nodeHit = raycaster.intersectObjects(networkHitTargets, false)[0];
      if (nodeHit) {
        const network = (currentPayload?.networks || []).find(item => Number(item.network_id) === nodeHit.object.userData.networkId);
        const node = (network?.nodes || []).find(item => Number(item.node_id) === nodeHit.object.userData.nodeId);
        if (node) {
          detail.innerHTML = `<b>Network N${network.network_id}</b> | Score ${Number(network.network_score || 0).toFixed(2)}<br>Nodes ${Number(network.node_count || 0)} | Edges ${Number(network.edge_count || 0)} | Age ${Number(network.age || 0).toLocaleString()}<br>Topology Persistence ${Number(network.topology_persistence || 0).toFixed(2)} | Node Stability ${Number(network.node_identity_stability || 0).toFixed(2)} | Edge Stability ${Number(network.edge_identity_stability || 0).toFixed(2)}<br><b>Node N${network.network_id}.${node.node_id}</b> | Particles ${Number(node.particle_count || 0).toLocaleString()} | Age ${Number(node.age || 0).toLocaleString()}<br>Density ${Number(node.density || 0).toFixed(3)} | Radius ${Number(node.radius || 0).toFixed(2)}<br>Velocity (${Number(node.mean_velocity?.[0] || 0).toFixed(2)}, ${Number(node.mean_velocity?.[1] || 0).toFixed(2)}) | Identity Score ${Number(node.identity_score || 0).toFixed(2)}<br>Species ${((node.species_distribution || []).map((value, index) => "S" + index + " " + (Number(value) * 100).toFixed(1) + "%").join(" | "))}`;
        }
        return;
      }
      const hit = raycaster.intersectObjects(clusterHitTargets, false)[0];
      if (hit) inspectCluster(hit.object.userData.clusterId);
    });
    renderer.domElement.addEventListener("wheel", event => { zoom = THREE.MathUtils.clamp(zoom + event.deltaY * 0.7, 500, 3200); event.preventDefault(); }, { passive: false });
    addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

    function animate() {
      requestAnimationFrame(animate);
      rotation += (targetRotation - rotation) * 0.08;
      world.rotation.y = rotation * 0.18;
      world.rotation.x = Math.sin(rotation * 0.7) * 0.035;
      camera.position.z += (zoom - camera.position.z) * 0.08;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>"""


class _ViewerHandler(BaseHTTPRequestHandler):
    server: "_ViewerHTTPServer"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path
        if route == "/":
            self._send(200, "text/html; charset=utf-8", VIEWER_HTML.encode("utf-8"))
            return
        if route == "/state":
            try:
                body = json.dumps(self.server.state_provider(), separators=(",", ":")).encode("utf-8")
            except Exception as error:  # keep the browser connected through a transient reset
                body = json.dumps({"error": str(error)}).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            # Browsers may cancel a poll while the simulator is closing.
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _ViewerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], provider: STATE_PROVIDER) -> None:
        self.state_provider = provider
        super().__init__(address, _ViewerHandler)


class ThreeViewerServer:
    """Serve the companion WebGL view on loopback without adding a GUI dependency."""

    def __init__(self, provider: STATE_PROVIDER) -> None:
        self.provider = provider
        self.server: _ViewerHTTPServer | None = None
        self.thread: Thread | None = None

    @property
    def url(self) -> str | None:
        if self.server is None:
            return None
        port = self.server.server_address[1]
        return f"http://127.0.0.1:{port}/"

    def start(self) -> str:
        if self.server is None:
            self.server = _ViewerHTTPServer(("127.0.0.1", 0), self.provider)
            self.thread = Thread(target=self.server.serve_forever, name="signal-three-viewer", daemon=True)
            self.thread.start()
        return self.url or "http://127.0.0.1/"

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.thread = None
