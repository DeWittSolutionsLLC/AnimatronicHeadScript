/**
 * neural.js — 3D knowledge graph using Three.js
 * Nodes = knowledge items, added incrementally as the AI learns.
 * New nodes grow from zero and pulse; existing nodes are never moved.
 */

(function () {
  const CATEGORIES = [
    { key: "quotes",       label: "Ultron Quotes", color: 0xff3333, pos: [ 0,  0,  0] },
    { key: "movie_quotes", label: "Movie Quotes",  color: 0xff7700, pos: [ 6,  2, -2] },
    { key: "song_quotes",  label: "Song Lyrics",   color: 0xcc44ff, pos: [-6,  2, -2] },
    { key: "traits",       label: "Traits",        color: 0x00aaff, pos: [ 4, -3,  3] },
    { key: "references",   label: "Slang",         color: 0x00ff99, pos: [-4, -3,  3] },
  ];

  const TOPICS_HUB_POS    = [0, 7, 0];
  const TOPICS_COLOR      = 0xffcc00;
  const _renderedTopicIds = new Set();

  // Self-generated hubs: key → { pos, color, itemSet }
  const _selfHubs = {};

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
  const starGeo   = new THREE.BufferGeometry();
  const starVerts = [];
  for (let i = 0; i < 1200; i++) {
    starVerts.push(
      (Math.random() - 0.5) * 120,
      (Math.random() - 0.5) * 120,
      (Math.random() - 0.5) * 120
    );
  }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starVerts, 3));
  scene.add(new THREE.Points(starGeo,
    new THREE.PointsMaterial({ color: 0x223344, size: 0.18 })));

  // Graph state
  const nodeObjects = {};  // id  → { mesh, pulseT, spawnT }
  const edges       = [];  // Three.Line objects (kept for future cleanup)
  const graphItems  = {};  // cat.key → Set<string> of item texts already in graph
  let   totalNodeCount = 0;

  function makeNodeMaterial(color, opacity) {
    opacity = opacity === undefined ? 1 : opacity;
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

  // Add an edge line between two positions
  function addEdge(posA, posB, color) {
    const points = [new THREE.Vector3(...posA), new THREE.Vector3(...posB)];
    const geo    = new THREE.BufferGeometry().setFromPoints(points);
    const mat    = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.18 });
    const line   = new THREE.Line(geo, mat);
    scene.add(line);
    edges.push(line);
  }

  /**
   * ensureItems — idempotent incremental update.
   * Adds any KB items not yet in the graph; skips existing ones.
   * New nodes spawn from scale 0 with a birth pulse.
   */
  function ensureItems(kb) {
    CATEGORIES.forEach(cat => {
      if (!graphItems[cat.key]) graphItems[cat.key] = new Set();
      const existing = graphItems[cat.key];

      // Hub node — created once, never moved
      const hubId = "cat_" + cat.key;
      if (!nodeObjects[hubId]) {
        const geo  = new THREE.SphereGeometry(0.45, 16, 16);
        const mesh = new THREE.Mesh(geo, makeNodeMaterial(cat.color));
        mesh.position.set(...cat.pos);
        mesh.userData = { label: cat.label, category: true, color: cat.color };
        scene.add(mesh);
        nodeObjects[hubId] = { mesh, pulseT: 0 };
      }

      const items    = (kb[cat.key] || []).filter(s => typeof s === "string" && s.trim());
      const newBatch = items.filter(t => !existing.has(t));
      if (!newBatch.length) return;

      // Radius grows with the total category size so nodes aren't packed too tight
      const clusterR = 2.2 + Math.min(items.length, 80) * 0.055;
      let catIdx = existing.size;

      newBatch.forEach(text => {
        existing.add(text);
        const pos      = scatter(cat.pos, clusterR);
        const size     = 0.09 + Math.min(text.length, 60) / 800;
        const geo2     = new THREE.SphereGeometry(size, 8, 8);
        const mat2     = makeNodeMaterial(cat.color, 0.85);
        const mesh2    = new THREE.Mesh(geo2, mat2);
        mesh2.position.set(...pos);
        const shortText = text.length > 80 ? text.slice(0, 77) + "..." : text;
        mesh2.userData  = { label: shortText, category: false, color: cat.color };
        mesh2.scale.setScalar(0);   // invisible until spawn animation runs
        scene.add(mesh2);

        const id = cat.key + "_" + catIdx++;
        // spawnT counts down 1→0 over ~0.5s, driving scale 0→1
        // pulseT gives a bright flash after the node reaches full size
        nodeObjects[id] = { mesh: mesh2, pulseT: 1.8, spawnT: 1.0 };
        totalNodeCount++;

        addEdge(pos, cat.pos, cat.color);
      });
    });

    document.getElementById("stat-nodes").textContent =
      totalNodeCount + " nodes";
    document.getElementById("stat-sessions").textContent =
      (kb.sessions || 0) + " sessions";
  }

  function ensureTopics(topics) {
    if (!Array.isArray(topics) || !topics.length) return;

    // Create the topics hub once
    if (!nodeObjects["cat_topics"]) {
      const geo  = new THREE.SphereGeometry(0.45, 16, 16);
      const mesh = new THREE.Mesh(geo, makeNodeMaterial(TOPICS_COLOR));
      mesh.position.set(...TOPICS_HUB_POS);
      mesh.userData = { label: "Discovered Topics", category: true, color: TOPICS_COLOR };
      scene.add(mesh);
      nodeObjects["cat_topics"] = { mesh, pulseT: 0 };
    }

    const newTopics = topics.filter(t => t.id && !_renderedTopicIds.has(t.id));
    if (!newTopics.length) return;

    const clusterR = 2.4 + Math.min(topics.length, 40) * 0.07;

    newTopics.forEach(function (topic) {
      _renderedTopicIds.add(topic.id);
      const pos  = scatter(TOPICS_HUB_POS, clusterR);
      const geo  = new THREE.SphereGeometry(0.22, 10, 10);
      const mat  = makeNodeMaterial(TOPICS_COLOR, 0.9);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...pos);

      const tip = topic.label + (topic.description ? "\n" + topic.description : "");
      mesh.userData = { label: tip, category: false, color: TOPICS_COLOR };
      mesh.scale.setScalar(0);
      scene.add(mesh);

      const id = "topics_" + topic.id;
      nodeObjects[id] = { mesh, pulseT: 1.8, spawnT: 1.0 };
      totalNodeCount++;

      addEdge(pos, TOPICS_HUB_POS, TOPICS_COLOR);
    });

    document.getElementById("stat-nodes").textContent = totalNodeCount + " nodes";
  }

  function ensureSelfHubs(hubs) {
    if (!Array.isArray(hubs) || !hubs.length) return;

    hubs.forEach(function (hub) {
      const key   = hub.key;
      const label = hub.label;
      const items = hub.items || [];
      const colorHex = parseInt((hub.color || "#ffffff").replace("#", ""), 16);

      // Spread new hubs in a ring around the scene, above the existing cluster
      if (!_selfHubs[key]) {
        const angle = Object.keys(_selfHubs).length * (Math.PI * 2 / 8);
        const r     = 9;
        const pos   = [
          r * Math.cos(angle),
          5 + (Math.random() - 0.5) * 2,
          r * Math.sin(angle),
        ];
        _selfHubs[key] = { pos, color: colorHex, itemSet: new Set() };

        // Hub sphere
        const geo  = new THREE.SphereGeometry(0.45, 16, 16);
        const mesh = new THREE.Mesh(geo, makeNodeMaterial(colorHex));
        mesh.position.set(...pos);
        mesh.userData = { label, category: true, color: colorHex };
        scene.add(mesh);
        nodeObjects["selfhub_" + key] = { mesh, pulseT: 2.0, spawnT: 1.0 };
        totalNodeCount++;

        // Connect new hub to origin
        addEdge(pos, [0, 0, 0], colorHex);

        // Add to legend dynamically
        const legend = document.getElementById("neural-legend");
        if (legend) {
          const item = document.createElement("div");
          item.className = "legend-item";
          item.innerHTML =
            `<span class="dot" style="background:${hub.color}"></span>${label}`;
          legend.appendChild(item);
        }
      }

      const hubData  = _selfHubs[key];
      const newItems = items.filter(function (i) { return !hubData.itemSet.has(i); });
      if (!newItems.length) return;

      const clusterR = 2.0 + Math.min(items.length, 40) * 0.06;

      newItems.forEach(function (text, idx) {
        hubData.itemSet.add(text);
        const pos2  = scatter(hubData.pos, clusterR);
        const size  = 0.11 + Math.min(text.length, 60) / 800;
        const geo2  = new THREE.SphereGeometry(size, 8, 8);
        const mat2  = makeNodeMaterial(hubData.color, 0.85);
        const mesh2 = new THREE.Mesh(geo2, mat2);
        mesh2.position.set(...pos2);
        const shortText = text.length > 80 ? text.slice(0, 77) + "..." : text;
        mesh2.userData  = { label: shortText, category: false, color: hubData.color };
        mesh2.scale.setScalar(0);
        scene.add(mesh2);

        const id = "selfhub_" + key + "_" + (hubData.itemSet.size + idx);
        nodeObjects[id] = { mesh: mesh2, pulseT: 1.8, spawnT: 1.0 };
        totalNodeCount++;

        addEdge(pos2, hubData.pos, hubData.color);
      });
    });

    document.getElementById("stat-nodes").textContent = totalNodeCount + " nodes";
  }

  // Public API
  window.neuralUpdate = function (kb) {
    ensureItems(kb);
    ensureTopics(kb.discovered_topics);
    ensureSelfHubs(kb.self_hubs);
  };

  // Pulse a category's existing nodes briefly (used by chat responses)
  window.neuralPulseCategory = function (catKey) {
    Object.entries(nodeObjects).forEach(function (entry) {
      var id = entry[0], obj = entry[1];
      if (id.startsWith(catKey) || id === "cat_" + catKey) {
        obj.pulseT = Math.max(obj.pulseT, 1.0);
      }
    });
  };

  // Raycaster for hover tooltip
  const raycaster = new THREE.Raycaster();
  const mouse     = new THREE.Vector2();
  const tooltip   = document.getElementById("node-tooltip");

  renderer.domElement.addEventListener("mousemove", function (e) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x =  ((e.clientX - rect.left)  / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)   / rect.height)  * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const meshes = Object.values(nodeObjects).map(function (n) { return n.mesh; });
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

    Object.values(nodeObjects).forEach(function (obj) {
      // Spawn animation: scale ramps from 0 → 1 as spawnT counts 1 → 0
      const spawning = obj.spawnT !== undefined && obj.spawnT > 0;
      if (spawning) {
        obj.spawnT = Math.max(0, obj.spawnT - dt * 2.5);
      }
      const spawnScale = spawning ? (1 - obj.spawnT) : 1;

      // Pulse: bright flash + over-scale
      if (obj.pulseT > 0) {
        obj.pulseT = Math.max(0, obj.pulseT - dt * 1.2);
        const t = Math.sin(obj.pulseT * Math.PI);
        obj.mesh.scale.setScalar(spawnScale * (1 + t * 0.6));
        obj.mesh.material.emissiveIntensity = 0.4 + t * 2.5;
      } else {
        obj.mesh.scale.setScalar(spawnScale);
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
  window.addEventListener("resize", function () {
    camera.aspect = W() / H();
    camera.updateProjectionMatrix();
    renderer.setSize(W(), H());
  });
})();
