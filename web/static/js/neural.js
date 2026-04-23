/**
 * neural.js — 3D knowledge graph using Three.js
 * Nodes = knowledge items, clustered by category, connected by lines.
 */

(function () {
  const CATEGORIES = [
    { key: "quotes",       label: "Ultron Quotes", color: 0xff3333, pos: [ 0,  0,  0] },
    { key: "movie_quotes", label: "Movie Quotes",  color: 0xff7700, pos: [ 6,  2, -2] },
    { key: "song_quotes",  label: "Song Lyrics",   color: 0xcc44ff, pos: [-6,  2, -2] },
    { key: "traits",       label: "Traits",        color: 0x00aaff, pos: [ 4, -3,  3] },
    { key: "references",   label: "Slang",         color: 0x00ff99, pos: [-4, -3,  3] },
  ];

  // Scene setup
  const container = document.getElementById("neural-container");
  const W = () => container.clientWidth;
  const H = () => container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W(), H());
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, W() / H(), 0.1, 200);
  camera.position.set(0, 4, 18);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping    = true;
  controls.dampingFactor    = 0.06;
  controls.autoRotate       = true;
  controls.autoRotateSpeed  = 0.4;
  controls.minDistance      = 6;
  controls.maxDistance      = 40;

  // Ambient + point lights
  scene.add(new THREE.AmbientLight(0x112233, 1.5));
  const redLight = new THREE.PointLight(0xff2222, 1.2, 40);
  redLight.position.set(0, 8, 0);
  scene.add(redLight);

  // Background particles
  const starGeo  = new THREE.BufferGeometry();
  const starVerts = [];
  for (let i = 0; i < 1200; i++) {
    starVerts.push((Math.random() - 0.5) * 120,
                   (Math.random() - 0.5) * 120,
                   (Math.random() - 0.5) * 120);
  }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starVerts, 3));
  scene.add(new THREE.Points(starGeo,
    new THREE.PointsMaterial({ color: 0x223344, size: 0.18 })));

  // Node/edge tracking
  const nodeObjects = {};  // id → { mesh, pulseT }
  const edges       = [];  // Line objects
  let   knowledgeData = null;

  function makeNodeMaterial(color, opacity = 1) {
    return new THREE.MeshPhongMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.4,
      transparent: opacity < 1,
      opacity,
    });
  }

  function scatter(center, radius) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = radius * (0.6 + Math.random() * 0.4);
    return [
      center[0] + r * Math.sin(phi) * Math.cos(theta),
      center[1] + r * Math.sin(phi) * Math.sin(theta),
      center[2] + r * Math.cos(phi),
    ];
  }

  function buildGraph(kb) {
    // Remove old nodes and edges
    Object.values(nodeObjects).forEach(n => scene.remove(n.mesh));
    edges.forEach(e => scene.remove(e));
    edges.length = 0;
    Object.keys(nodeObjects).forEach(k => delete nodeObjects[k]);

    const catPositions = {};
    let totalNodes = 0;

    CATEGORIES.forEach(cat => {
      const items = (kb[cat.key] || []).filter(s => typeof s === "string" && s.trim());

      // Category hub node (larger)
      const geo  = new THREE.SphereGeometry(0.45, 16, 16);
      const mat  = makeNodeMaterial(cat.color);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...cat.pos);
      mesh.userData = { label: cat.label, category: true, color: cat.color };
      scene.add(mesh);
      nodeObjects[`cat_${cat.key}`] = { mesh, pulseT: 0 };
      catPositions[cat.key] = cat.pos;

      // Item nodes
      const clusterR = 2.0 + Math.min(items.length, 20) * 0.08;
      items.slice(0, 22).forEach((text, i) => {
        const pos  = scatter(cat.pos, clusterR);
        const size = 0.13 + Math.min(text.length, 60) / 600;
        const g2   = new THREE.SphereGeometry(size, 8, 8);
        const m2   = makeNodeMaterial(cat.color, 0.85);
        const mesh2 = new THREE.Mesh(g2, m2);
        mesh2.position.set(...pos);
        const shortText = text.length > 80 ? text.slice(0, 77) + "..." : text;
        mesh2.userData = { label: shortText, category: false, color: cat.color };
        scene.add(mesh2);
        const id = `${cat.key}_${i}`;
        nodeObjects[id] = { mesh: mesh2, pulseT: 0 };
        totalNodes++;

        // Edge: item → category hub
        const points = [
          new THREE.Vector3(...pos),
          new THREE.Vector3(...cat.pos),
        ];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        const lineMat = new THREE.LineBasicMaterial({
          color: cat.color,
          transparent: true,
          opacity: 0.18,
        });
        const line = new THREE.Line(lineGeo, lineMat);
        scene.add(line);
        edges.push(line);
      });
    });

    // Update stats
    document.getElementById("stat-nodes").textContent =
      `${totalNodes} nodes`;
    document.getElementById("stat-sessions").textContent =
      `${kb.sessions || 0} sessions`;
  }

  // Public: call when knowledge changes
  window.neuralUpdate = function (kb) {
    knowledgeData = kb;
    buildGraph(kb);
  };

  // Public: pulse a category's nodes briefly
  window.neuralPulseCategory = function (catKey) {
    Object.entries(nodeObjects).forEach(([id, obj]) => {
      if (id.startsWith(catKey) || id === `cat_${catKey}`) {
        obj.pulseT = 1.0;
      }
    });
  };

  // Raycaster for hover tooltip
  const raycaster = new THREE.Raycaster();
  const mouse     = new THREE.Vector2();
  const tooltip   = document.getElementById("node-tooltip");

  renderer.domElement.addEventListener("mousemove", e => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x =  ((e.clientX - rect.left)  / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)   / rect.height)  * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const meshes = Object.values(nodeObjects).map(n => n.mesh);
    const hits   = raycaster.intersectObjects(meshes);
    if (hits.length) {
      tooltip.textContent = hits[0].object.userData.label || "";
      tooltip.classList.add("visible");
    } else {
      tooltip.classList.remove("visible");
    }
  });

  // Animation loop
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();

    // Pulse effect: scale + emissive boost
    Object.values(nodeObjects).forEach(obj => {
      if (obj.pulseT > 0) {
        obj.pulseT = Math.max(0, obj.pulseT - dt * 1.2);
        const t = Math.sin(obj.pulseT * Math.PI);
        obj.mesh.scale.setScalar(1 + t * 0.6);
        obj.mesh.material.emissiveIntensity = 0.4 + t * 2.5;
      } else {
        obj.mesh.scale.setScalar(1);
        obj.mesh.material.emissiveIntensity = 0.4;
      }
    });

    // Gently bob the red point light
    const t = performance.now() / 1000;
    redLight.position.y = 8 + Math.sin(t * 0.7) * 2;

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize
  window.addEventListener("resize", () => {
    camera.aspect = W() / H();
    camera.updateProjectionMatrix();
    renderer.setSize(W(), H());
  });
})();
