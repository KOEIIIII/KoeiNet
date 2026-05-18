(() => {
  const videoName = window.VALIDATION_VIDEO_NAME;
  const TEXT = {
    saveOk: "保存成功",
    saveAuto: "已自动保存",
    saveFail: "保存失败",
    loading: "正在加载…",
    loaded: "已加载",
    unsaved: "当前修改尚未保存",
    reviewed: "状态：已评分",
    unreviewed: "状态：未评分",
    seeked: "已回到当前片段起点",
    noFrames: "当前片段没有可用的代表性画面。",
    noEvidence: "当前没有可展示的证据摘要。",
  };
  const FIELD_ZH = {
    segment_id: "片段编号",
    start_time_sec: "开始时间（秒）",
    end_time_sec: "结束时间（秒）",
    center_time_sec: "中心时间（秒）",
    included_frame_count: "纳入帧数",
    reviewed: "是否已评分",
    matched_gps_longitude_gcj02: "匹配经度 GCJ-02",
    matched_gps_latitude_gcj02: "匹配纬度 GCJ-02",
    matched_gps_longitude_wgs84: "匹配经度 WGS84",
    matched_gps_latitude_wgs84: "匹配纬度 WGS84",
    segment_center_time_utc: "片段中心 UTC",
    match_status: "匹配状态",
    confidence: "匹配置信度",
    vis_green_pixel_ratio_mean: "绿色像素比例",
    vis_sky_blue_ratio_top_mean: "天空蓝占比",
    vis_brightness_mean_mean: "平均亮度",
    vis_edge_density_mean: "边缘密度",
    ai_activity_major_label: "主要视觉活动",
    ai_activity_suitable_labels: "适合活动",
    top_k_events: "主要声音事件",
    event_class_distribution_json: "声音事件分布",
    group_ratio_traffic: "交通声占比",
    group_ratio_human: "人声占比",
    group_ratio_nature: "自然声占比",
    group_ratio_mechanical: "机械声占比",
    audio_signal__loudness_proxy_db: "响度代理值",
    visual_semantic__road__mean: "道路语义占比",
    visual_semantic__sidewalk__mean: "人行道语义占比",
    visual_semantic__building__mean: "建筑语义占比",
    green_view__greenviewindex__mean: "绿视率",
    people__total_people__mean: "平均人数",
    emotion__beautiful__mean: "美感倾向",
    emotion__depressing__mean: "压抑感倾向",
    cross_modal_reason: "街景-声景关系解释",
    problem_labels: "问题标签",
    severity_scores: "严重度分项",
    priority_actions: "建议行动",
    status: "诊断状态",
    sound_top_events: "主要声音事件",
    traffic_sound_ratio: "交通声占比",
    natural_sound_ratio: "自然声占比",
    road_ratio: "道路占比",
    sidewalk_ratio: "人行道占比",
    people_mean: "平均人数",
    green_ratio: "绿色比例",
    sky_ratio: "天空比例",
  };
  const STREET_ZH = {
    mixed_use: "混合用地",
    commercial: "商业街区",
    residential: "居住街区",
    arterial: "干道",
    park: "公园",
    campus: "校园",
  };
  const LABEL_ZH = {
    traffic_noise: "交通噪声",
    pedestrian_discomfort: "步行不适",
    visual_clutter: "视觉杂乱",
    low_green_view: "绿视率偏低",
    high_hardscape: "硬质铺装偏高",
    vehicle_dominance: "车辆主导",
    poor_walkability_cues: "步行友好线索不足",
    low_aesthetic_quality: "审美质量偏低",
    high_loudness: "响度过高",
    low_natural_sound: "自然声不足",
    human_voice_dominant: "人声占比过高",
    high_eventfulness: "事件性过高",
    low_eventfulness: "事件性过低",
    noisy_but_low_pleasantness: "噪声高且愉悦度低",
    no_major_problem: "无主要问题",
  };

  class ValidationApp {
    constructor(videoName) {
      this.videoName = videoName;
      this.payload = null;
      this.segments = [];
      this.filteredIndices = [];
      this.currentFilteredPosition = 0;
      this.options = { street_type: [], problem_labels: [], score_values: [1, 2, 3, 4, 5] };
      this.formDirty = false;
      this.mediaStopAt = null;
      this.audioStopAt = null;
      this.dom = {};
    }

    async init() {
      this.cacheDom();
      this.bindEvents();
      this.initScoreGroups();
      this.populateStreetTypes();
      await this.loadPayload();
    }

    cacheDom() {
      const ids = {
        media: "segment-media",
        audio: "segment-audio",
        segmentSelector: "segment-selector",
        filterUnreviewed: "filter-unreviewed-toggle",
        progress: "validation-progress",
        saveStatus: "save-status",
        segmentTime: "segment-time-pill",
        segmentCounter: "segment-counter-pill",
        segmentState: "segment-state-pill",
        bootstrapPill: "bootstrap-pill",
        frameCount: "frame-count-pill",
        frameStrip: "frame-preview-strip",
        segmentMeta: "segment-meta",
        segmentGeo: "segment-geo",
        segmentVisual: "segment-visual",
        segmentSoundscape: "segment-soundscape",
        segmentFusion: "segment-fusion",
        segmentRelationship: "segment-relationship",
        segmentDiagnostics: "segment-diagnostics",
        segmentId: "segment-id-input",
        streetType: "street-type-select",
        primaryProblemLabel: "primary-problem-label-select",
        annotatorNotes: "annotator-notes-textarea",
        saveButton: "save-segment-btn",
        prevButton: "prev-segment-btn",
        nextButton: "next-segment-btn",
        playButton: "play-segment-btn",
        playAudioButton: "play-audio-btn",
        seekButton: "seek-segment-btn",
      };
      Object.entries(ids).forEach(([key, id]) => { this.dom[key] = document.getElementById(id); });
    }

    bindEvents() {
      this.dom.segmentSelector.addEventListener("change", async (event) => {
        await this.selectBySegmentId(Number(event.target.value));
      });
      this.dom.filterUnreviewed.addEventListener("change", async () => {
        if (!(await this.saveIfDirty())) {
          this.dom.filterUnreviewed.checked = !this.dom.filterUnreviewed.checked;
          return;
        }
        this.rebuildFilteredIndices();
        this.buildSegmentSelector();
        this.renderCurrentSegment();
      });
      this.dom.saveButton.addEventListener("click", async () => this.saveCurrentSegment());
      this.dom.prevButton.addEventListener("click", async () => this.navigate(-1));
      this.dom.nextButton.addEventListener("click", async () => this.navigate(1));
      this.dom.playButton.addEventListener("click", () => this.playCurrentSegmentClip());
      this.dom.playAudioButton.addEventListener("click", () => this.playCurrentSegmentAudio());
      this.dom.seekButton.addEventListener("click", () => this.seekCurrentSegmentStart(true));
      this.dom.media.addEventListener("timeupdate", () => {
        if (this.mediaStopAt != null && this.dom.media.currentTime >= this.mediaStopAt) this.dom.media.pause();
      });
      this.dom.audio.addEventListener("timeupdate", () => {
        if (this.audioStopAt != null && this.dom.audio.currentTime >= this.audioStopAt) this.dom.audio.pause();
      });
      window.addEventListener("beforeunload", (event) => {
        if (!this.formDirty) return;
        event.preventDefault();
        event.returnValue = "当前片段有未保存修改，离开页面会丢失。";
      });
      document.addEventListener("keydown", async (event) => {
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
        if (event.ctrlKey && event.key.toLowerCase() === "s") {
          event.preventDefault();
          await this.saveCurrentSegment();
          return;
        }
        if (activeTag === "textarea") return;
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          await this.navigate(-1);
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          await this.navigate(1);
        }
      });
    }

    initScoreGroups() {
      ["comfort_score", "vitality_score", "soundscape_pleasantness", "soundscape_eventfulness", "overall_problem_severity", "confidence_score"]
        .forEach((fieldName) => {
          const container = document.getElementById(`${fieldName.replaceAll("_", "-")}-group`);
          container.innerHTML = "";
          [1, 2, 3, 4, 5].forEach((score) => {
            const label = document.createElement("label");
            label.className = "score-pill";
            label.innerHTML = `<input type="radio" name="${fieldName}" value="${score}"><span>${score}</span>`;
            label.querySelector("input").addEventListener("change", () => this.markDirty());
            container.appendChild(label);
          });
        });
    }

    populateStreetTypes() {
      this.dom.streetType.innerHTML = "";
      this.options.street_type.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = STREET_ZH[value] || value;
        this.dom.streetType.appendChild(option);
      });
      this.dom.streetType.addEventListener("change", () => this.markDirty());
      this.dom.annotatorNotes.addEventListener("input", () => this.markDirty());
    }

    async loadPayload() {
      this.setSaveStatus(TEXT.loading, "muted");
      const response = await fetch(`/api/validation/${this.videoName}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "评分数据加载失败");
      this.payload = payload;
      this.segments = Array.isArray(payload.segments) ? payload.segments : [];
      this.options = payload.options || this.options;
      this.populateStreetTypes();
      this.rebuildFilteredIndices();
      this.buildSegmentSelector();
      this.setSaveStatus(TEXT.loaded, "success");
      this.renderCurrentSegment();
    }

    rebuildFilteredIndices() {
      const onlyUnreviewed = this.dom.filterUnreviewed.checked;
      this.filteredIndices = this.segments.map((segment, index) => ({ segment, index }))
        .filter(({ segment }) => !onlyUnreviewed || !segment.reviewed)
        .map(({ index }) => index);
      if (!this.filteredIndices.length) this.filteredIndices = this.segments.map((_, index) => index);
      if (this.currentFilteredPosition >= this.filteredIndices.length) this.currentFilteredPosition = 0;
      this.updateProgress();
    }

    buildSegmentSelector() {
      const currentSegmentId = this.currentSegment()?.segment_id;
      this.dom.segmentSelector.innerHTML = "";
      this.filteredIndices.forEach((segmentIndex, filteredPosition) => {
        const segment = this.segments[segmentIndex];
        const option = document.createElement("option");
        option.value = segment.segment_id;
        option.textContent = `片段 #${segment.segment_id} · ${segment.start_time_sec}s - ${segment.end_time_sec}s${segment.reviewed ? " · 已评分" : " · 未评分"}`;
        this.dom.segmentSelector.appendChild(option);
        if (currentSegmentId != null && segment.segment_id === currentSegmentId) this.currentFilteredPosition = filteredPosition;
      });
    }

    currentSegment() {
      if (!this.filteredIndices.length) return null;
      return this.segments[this.filteredIndices[this.currentFilteredPosition] ?? 0] || null;
    }

    async selectBySegmentId(segmentId) {
      const position = this.filteredIndices.findIndex((index) => this.segments[index].segment_id === segmentId);
      if (position === -1) return;
      if (!(await this.saveIfDirty())) {
        this.syncSelectorToCurrent();
        return;
      }
      this.currentFilteredPosition = position;
      this.renderCurrentSegment();
    }

    async navigate(delta) {
      if (!this.filteredIndices.length) return;
      if (!(await this.saveIfDirty())) return;
      const next = this.currentFilteredPosition + delta;
      if (next < 0 || next >= this.filteredIndices.length) return;
      this.currentFilteredPosition = next;
      this.renderCurrentSegment();
    }

    async saveIfDirty() {
      if (!this.formDirty) return true;
      const shouldSave = window.confirm("当前片段有未保存修改。点击“确定”先保存并继续切换；点击“取消”则留在当前片段。");
      if (!shouldSave) return false;
      return await this.saveCurrentSegment(true);
    }

    collectFormState() {
      const segment = this.currentSegment();
      const mainLabels = Array.from(document.querySelectorAll('input[name="main_problem_labels"]:checked')).map((input) => input.value);
      return {
        segment_id: segment.segment_id,
        street_type: this.dom.streetType.value,
        comfort_score: this.getRadioValue("comfort_score"),
        vitality_score: this.getRadioValue("vitality_score"),
        soundscape_pleasantness: this.getRadioValue("soundscape_pleasantness"),
        soundscape_eventfulness: this.getRadioValue("soundscape_eventfulness"),
        overall_problem_severity: this.getRadioValue("overall_problem_severity"),
        confidence_score: this.getRadioValue("confidence_score"),
        main_problem_labels: mainLabels,
        primary_problem_label: this.dom.primaryProblemLabel.value,
        annotator_notes: this.dom.annotatorNotes.value.trim(),
      };
    }

    getRadioValue(fieldName) {
      const checked = document.querySelector(`input[name="${fieldName}"]:checked`);
      return checked ? Number(checked.value) : null;
    }

    async saveCurrentSegment(quiet = false) {
      const segment = this.currentSegment();
      if (!segment) return false;
      const payload = this.collectFormState();
      const savedSegmentId = segment.segment_id;
      this.setSaveStatus("正在保存…", "muted");
      const response = await fetch(`/api/validation/${this.videoName}/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) {
        this.setSaveStatus(result.error || TEXT.saveFail, "error");
        return false;
      }
      segment.validation = { ...segment.validation, ...payload, main_problem_labels: [...payload.main_problem_labels] };
      segment.reviewed = this.isReviewedPayload(payload);
      this.formDirty = false;
      this.rebuildFilteredIndices();
      this.buildSegmentSelector();
      const visiblePosition = this.filteredIndices.findIndex((index) => this.segments[index].segment_id === savedSegmentId);
      if (visiblePosition >= 0) this.currentFilteredPosition = visiblePosition;
      this.renderCurrentSegment();
      this.setSaveStatus(quiet ? TEXT.saveAuto : TEXT.saveOk, "success");
      return true;
    }

    isReviewedPayload(payload) {
      const values = [payload.comfort_score, payload.vitality_score, payload.soundscape_pleasantness, payload.soundscape_eventfulness, payload.overall_problem_severity];
      return values.every((v) => Number.isInteger(v) && v >= 1 && v <= 5) && Boolean(payload.street_type) && Boolean(payload.primary_problem_label);
    }

    hasBootstrapDraft(validation) {
      const labels = Array.isArray(validation.main_problem_labels) ? validation.main_problem_labels : [];
      return Boolean(labels.length || (validation.primary_problem_label && validation.primary_problem_label !== "no_major_problem") || Number.isInteger(validation.overall_problem_severity) || Number.isInteger(validation.comfort_score) || Number.isInteger(validation.vitality_score) || Number.isInteger(validation.soundscape_pleasantness) || Number.isInteger(validation.soundscape_eventfulness));
    }

    setSaveStatus(text, variant) {
      this.dom.saveStatus.textContent = text;
      this.dom.saveStatus.dataset.variant = variant || "muted";
    }

    markDirty() {
      this.formDirty = true;
      this.setSaveStatus(TEXT.unsaved, "warning");
    }

    syncSelectorToCurrent() {
      const segment = this.currentSegment();
      if (segment) this.dom.segmentSelector.value = String(segment.segment_id);
    }

    updateProgress() {
      const summary = this.payload ? this.payload.summary : { total_segments: 0, reviewed_segments: 0, unreviewed_segments: 0 };
      this.dom.progress.textContent = `已评分 ${summary.reviewed_segments} / ${summary.total_segments} · 未评分 ${summary.unreviewed_segments}`;
    }

    renderCurrentSegment() {
      const segment = this.currentSegment();
      if (!segment) return;
      this.syncSelectorToCurrent();
      this.payload.summary.reviewed_segments = this.segments.filter((item) => item.reviewed).length;
      this.payload.summary.total_segments = this.segments.length;
      this.payload.summary.unreviewed_segments = this.segments.length - this.payload.summary.reviewed_segments;
      this.updateProgress();
      this.dom.segmentId.value = segment.segment_id;
      this.dom.streetType.value = segment.validation.street_type || "mixed_use";
      this.setRadioValue("comfort_score", segment.validation.comfort_score);
      this.setRadioValue("vitality_score", segment.validation.vitality_score);
      this.setRadioValue("soundscape_pleasantness", segment.validation.soundscape_pleasantness);
      this.setRadioValue("soundscape_eventfulness", segment.validation.soundscape_eventfulness);
      this.setRadioValue("overall_problem_severity", segment.validation.overall_problem_severity);
      this.setRadioValue("confidence_score", segment.validation.confidence_score);
      this.dom.annotatorNotes.value = segment.validation.annotator_notes || "";
      this.renderProblemLabelSelector(segment.validation.main_problem_labels || []);
      this.renderPrimaryProblemOptions(segment.validation.main_problem_labels || [], segment.validation.primary_problem_label);
      this.renderFrameStrip(segment);
      this.renderMeta(segment);
      this.renderEvidenceList(this.dom.segmentGeo, segment.geo_summary);
      this.renderEvidenceList(this.dom.segmentVisual, segment.visual_summary);
      this.renderEvidenceList(this.dom.segmentSoundscape, segment.soundscape_summary);
      this.renderEvidenceList(this.dom.segmentFusion, segment.fusion_summary);
      this.renderEvidenceList(this.dom.segmentRelationship, segment.relationship_summary);
      this.renderDiagnostics(segment.diagnostics_summary || {});
      this.setSegmentMedia(segment);
      this.setSegmentAudio(segment);
      this.dom.segmentCounter.textContent = `片段 ${this.currentFilteredPosition + 1} / ${this.filteredIndices.length}`;
      this.dom.segmentState.textContent = segment.reviewed ? TEXT.reviewed : TEXT.unreviewed;
      this.dom.segmentState.dataset.variant = segment.reviewed ? "success" : "warning";
      this.dom.bootstrapPill.textContent = this.hasBootstrapDraft(segment.validation) ? "初稿状态：已载入可复核初值" : "初稿状态：当前为空白";
      this.dom.bootstrapPill.dataset.variant = this.hasBootstrapDraft(segment.validation) ? "success" : "muted";
      this.formDirty = false;
      this.setSaveStatus(segment.reviewed ? "已加载该片段评分" : "该片段尚未完成评分", segment.reviewed ? "success" : "muted");
    }

    setRadioValue(name, value) {
      document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
        input.checked = Number(input.value) === Number(value);
      });
    }

    renderProblemLabelSelector(selectedLabels) {
      const container = document.getElementById("main-problem-labels-group");
      container.innerHTML = "";
      this.options.problem_labels.forEach((value) => {
        const checked = selectedLabels.includes(value) ? "checked" : "";
        const item = document.createElement("label");
        item.className = "tag-chip";
        item.innerHTML = `<input type="checkbox" name="main_problem_labels" value="${value}" ${checked}><span>${LABEL_ZH[value] || value}</span>`;
        item.querySelector("input").addEventListener("change", () => {
          this.markDirty();
          const labels = Array.from(document.querySelectorAll('input[name="main_problem_labels"]:checked')).map((input) => input.value);
          this.renderPrimaryProblemOptions(labels, this.dom.primaryProblemLabel.value);
        });
        container.appendChild(item);
      });
    }

    renderPrimaryProblemOptions(selectedLabels, currentValue) {
      const options = ["no_major_problem", ...selectedLabels.filter((label, index) => selectedLabels.indexOf(label) === index)];
      const fallback = options.includes(currentValue) ? currentValue : "no_major_problem";
      this.dom.primaryProblemLabel.innerHTML = "";
      options.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = LABEL_ZH[value] || value;
        this.dom.primaryProblemLabel.appendChild(option);
      });
      this.dom.primaryProblemLabel.value = fallback;
      this.dom.primaryProblemLabel.onchange = () => this.markDirty();
    }

    renderFrameStrip(segment) {
      const urls = [];
      if (segment.frame_strip_url) urls.push({ url: segment.frame_strip_url, label: "上下文拼图" });
      if (segment.primary_preview_url) urls.push({ url: segment.primary_preview_url, label: "主预览图" });
      (segment.frame_preview_urls || []).forEach((url, index) => {
        const labels = ["起始画面", "中间画面", "结束画面"];
        urls.push({ url, label: labels[index] || `画面 ${index + 1}` });
      });
      this.dom.frameStrip.innerHTML = "";
      this.dom.frameCount.textContent = `${urls.length} 张代表性画面`;
      if (!urls.length) {
        this.dom.frameStrip.innerHTML = `<div class="empty-hint">${TEXT.noFrames}</div>`;
        return;
      }
      urls.forEach((item) => {
        const card = document.createElement("div");
        card.className = "frame-preview-card";
        card.innerHTML = `<img src="${item.url}" alt="${item.label}"><span>${item.label}</span>`;
        this.dom.frameStrip.appendChild(card);
      });
    }

    renderMeta(segment) {
      this.dom.segmentTime.textContent = `${segment.start_time_sec ?? "--"} 秒 → ${segment.end_time_sec ?? "--"} 秒`;
      this.renderEvidenceList(this.dom.segmentMeta, {
        segment_id: segment.segment_id,
        start_time_sec: segment.start_time_sec,
        end_time_sec: segment.end_time_sec,
        center_time_sec: segment.center_time_sec,
        included_frame_count: segment.included_frame_count,
        reviewed: segment.reviewed ? "是" : "否",
      });
    }

    renderEvidenceList(container, data) {
      container.innerHTML = "";
      const entries = Object.entries(data || {}).filter(([, value]) => value !== null && value !== undefined && value !== "");
      if (!entries.length) {
        container.innerHTML = `<div class="empty-hint">${TEXT.noEvidence}</div>`;
        return;
      }
      entries.forEach(([key, value]) => {
        const row = document.createElement("div");
        row.className = "evidence-row";
        row.innerHTML = `<span class="evidence-key">${FIELD_ZH[key] || key}</span><span class="evidence-value">${this.formatValue(value)}</span>`;
        container.appendChild(row);
      });
    }

    renderDiagnostics(data) {
      this.renderEvidenceList(this.dom.segmentDiagnostics, {
        problem_labels: Array.isArray(data.problem_labels) ? data.problem_labels : [],
        severity_scores: data.severity_scores || {},
        cross_modal_reason: data.cross_modal_reason || null,
        priority_actions: Array.isArray(data.priority_actions) ? data.priority_actions : [],
        status: data.status || null,
      });
    }

    formatValue(value) {
      if (Array.isArray(value)) return value.length ? value.map((item) => LABEL_ZH[String(item)] || String(item)).join("、") : "无";
      if (value && typeof value === "object") return JSON.stringify(value);
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
      return String(value);
    }

    setSegmentMedia(segment) {
      const poster = segment.primary_preview_url || (segment.frame_preview_urls && segment.frame_preview_urls[1]) || (segment.frame_preview_urls && segment.frame_preview_urls[0]) || "";
      if (poster) this.dom.media.poster = poster;
      if (this.dom.media.dataset.videoUrl !== segment.video_url) {
        this.dom.media.pause();
        this.dom.media.removeAttribute("src");
        this.dom.media.src = segment.video_url;
        this.dom.media.dataset.videoUrl = segment.video_url;
        this.dom.media.load();
      }
      this.dom.media.pause();
      this.mediaStopAt = segment.end_time_sec ?? null;
      this.seekMedia(segment.start_time_sec, true);
    }

    setSegmentAudio(segment) {
      if (!segment.audio_url) {
        this.dom.audio.pause();
        this.dom.audio.removeAttribute("src");
        this.dom.audio.load();
        return;
      }
      if (this.dom.audio.dataset.audioUrl !== segment.audio_url) {
        this.dom.audio.pause();
        this.dom.audio.removeAttribute("src");
        this.dom.audio.src = segment.audio_url;
        this.dom.audio.dataset.audioUrl = segment.audio_url;
        this.dom.audio.dataset.audioMode = segment.audio_mode || "clip";
        this.dom.audio.load();
      }
      if ((segment.audio_mode || "clip") === "clip") {
        this.audioStopAt = null;
        this.seekAudio(0);
      } else {
        this.audioStopAt = segment.end_time_sec ?? null;
        this.seekAudio(segment.start_time_sec);
      }
    }

    playCurrentSegmentClip() {
      const segment = this.currentSegment();
      if (!segment) return;
      this.mediaStopAt = segment.end_time_sec ?? null;
      this.seekMedia(segment.start_time_sec, true, () => this.dom.media.play().catch(() => {}));
    }

    playCurrentSegmentAudio() {
      const segment = this.currentSegment();
      if (!segment || !segment.audio_url) return;
      if ((segment.audio_mode || "clip") === "clip") {
        this.audioStopAt = null;
        this.seekAudio(0, () => this.dom.audio.play().catch(() => {}));
      } else {
        this.audioStopAt = segment.end_time_sec ?? null;
        this.seekAudio(segment.start_time_sec, () => this.dom.audio.play().catch(() => {}));
      }
    }

    seekCurrentSegmentStart(showStatus = false) {
      const segment = this.currentSegment();
      if (!segment) return;
      this.dom.media.pause();
      this.mediaStopAt = segment.end_time_sec ?? null;
      this.seekMedia(segment.start_time_sec, true);
      if (showStatus) this.setSaveStatus(TEXT.seeked, "muted");
    }

    seekMedia(timeSec, syncLabel = false, callback = null) {
      if (timeSec == null || !Number.isFinite(timeSec)) return;
      const applySeek = () => {
        try {
          this.dom.media.currentTime = timeSec;
          if (syncLabel) {
            const segment = this.currentSegment();
            if (segment) this.dom.segmentTime.textContent = `${segment.start_time_sec ?? "--"} 秒 → ${segment.end_time_sec ?? "--"} 秒`;
          }
          if (typeof callback === "function") callback();
        } catch (error) {
          console.warn("视频定位失败：", error);
        }
      };
      if (this.dom.media.readyState >= 1) applySeek();
      else this.dom.media.addEventListener("loadedmetadata", applySeek, { once: true });
    }

    seekAudio(timeSec, callback = null) {
      if (timeSec == null || !Number.isFinite(timeSec)) return;
      const applySeek = () => {
        try {
          this.dom.audio.currentTime = timeSec;
          if (typeof callback === "function") callback();
        } catch (error) {
          console.warn("音频定位失败：", error);
        }
      };
      if (this.dom.audio.readyState >= 1) applySeek();
      else this.dom.audio.addEventListener("loadedmetadata", applySeek, { once: true });
    }
  }

  window.addEventListener("DOMContentLoaded", async () => {
    const app = new ValidationApp(videoName);
    try {
      await app.init();
    } catch (error) {
      console.error(error);
      const saveStatus = document.getElementById("save-status");
      saveStatus.textContent = error.message || "评分页面加载失败";
      saveStatus.dataset.variant = "error";
    }
  });
})();
