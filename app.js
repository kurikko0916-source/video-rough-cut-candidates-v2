const $ = selector => document.querySelector(selector);
const state = { file: null, jobId: null, result: null, pollTimer: null };
const views = { 1: $('#select-view'), 2: $('#analyze-view'), 3: $('#result-view') };
const phaseProgress = { queued: 5, upload: 12, probing: 20, audio: 33, transcribe: 58, analyze: 78, merge: 92, complete: 100 };
const phaseMessages = {
  queued: '処理の準備をしています…', upload: '動画をMac内に読み込みました', probing: '動画情報を確認しています…',
  audio: '文字起こし用の音声を取り出しています…', transcribe: '日本語の会話を文字起こししています…',
  analyze: '言い直し・仕切り直し・無音を確認しています…', merge: '発言の境界に合わせて候補を整えています…', complete: '解析が完了しました'
};

function toast(message) { const el = $('#toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove('show'), 2200); }
function showStep(step) {
  Object.entries(views).forEach(([key, view]) => view.classList.toggle('active-view', Number(key) === step));
  document.querySelectorAll('.step').forEach(el => { const value = Number(el.dataset.step); el.classList.toggle('active', value === step); el.classList.toggle('done', value < step); });
}
function formatBytes(bytes) { if (!bytes) return '0 MB'; const mb = bytes / 1024 / 1024; return `${mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb.toFixed(1) + ' MB'}`; }
function formatClock(seconds) { const safe = Math.max(0, Number(seconds) || 0); const h = Math.floor(safe / 3600); const m = Math.floor((safe % 3600) / 60); const s = Math.floor(safe % 60); return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

function chooseFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mp4') && file.type !== 'video/mp4') return toast('MP4動画を選んでください');
  state.file = file; $('#file-name').textContent = file.name; $('#file-meta').textContent = `${formatBytes(file.size)} ・ MP4`;
  $('#file-card').classList.remove('hidden'); $('#start-button').disabled = false;
  const video = document.createElement('video'); video.preload = 'metadata'; video.onloadedmetadata = () => { $('#file-meta').textContent = `${formatClock(video.duration)} ・ ${formatBytes(file.size)} ・ MP4`; URL.revokeObjectURL(video.src); }; video.src = URL.createObjectURL(file);
}
function clearFile() { state.file = null; $('#video-input').value = ''; $('#file-card').classList.add('hidden'); $('#start-button').disabled = true; }

async function checkEnvironment(showDialog = false) {
  try {
    const response = await fetch('/api/health'); const data = await response.json(); const tools = data.tools || {};
    const requiredReady = tools.ffmpeg?.ready && tools.ffprobe?.ready && tools.whisper?.ready && tools.whisper_model?.ready;
    $('#environment-dot').className = `status-dot ${requiredReady ? 'ready' : 'error'}`;
    if (showDialog) {
      const labels = { ffmpeg:'FFmpeg（音声抽出）', ffprobe:'ffprobe（動画情報）', whisper:'whisper.cpp（文字起こし）', whisper_model:'Whisperモデル', ollama:'Ollama（内容判定）' };
      $('#environment-list').innerHTML = Object.entries(labels).map(([key,label]) => `<div class="environment-item"><strong>${label}</strong><span class="${tools[key]?.ready ? 'ok' : 'missing'}">${tools[key]?.ready ? '準備完了' : '未セットアップ'}</span></div>`).join('');
      $('#environment-dialog').showModal();
    }
    return requiredReady;
  } catch { $('#environment-dot').className = 'status-dot error'; if (showDialog) toast('ローカル処理に接続できません'); return false; }
}

async function startAnalysis() {
  if (!state.file) return;
  const ready = await checkEnvironment(false);
  if (!ready) { await checkEnvironment(true); return toast('必要なローカルツールを準備してください'); }
  showStep(2); setProgress('upload', 4);
  try {
    const response = await fetch('/api/jobs', { method:'POST', headers:{'Content-Type':'video/mp4','X-Filename':encodeURIComponent(state.file.name),'X-File-Size':String(state.file.size)}, body:state.file });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || '動画を読み込めませんでした');
    state.jobId = data.job_id; pollJob();
  } catch (error) { showStep(1); toast(error.message); }
}
function setProgress(phase, serverProgress) {
  const progress = Number(serverProgress ?? phaseProgress[phase] ?? 0); $('#progress-bar').style.width = `${progress}%`; $('#progress-percent').textContent = `${progress}%`; $('#progress-message').textContent = phaseMessages[phase] || '処理中です…';
  const order = ['upload','audio','transcribe','analyze']; const current = Math.max(0, order.indexOf(phase === 'probing' ? 'upload' : phase === 'merge' ? 'analyze' : phase));
  document.querySelectorAll('.process-list span').forEach((el,index) => { el.classList.toggle('active', index === current); el.classList.toggle('done', index < current || phase === 'complete'); });
}
async function pollJob() {
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`); const data = await response.json(); if (!response.ok) throw new Error(data.error || '状態を取得できません');
    setProgress(data.phase, data.progress);
    if (data.status === 'complete') { state.result = data; renderResults(); return; }
    if (data.status === 'failed') throw new Error(data.error || '解析に失敗しました');
    state.pollTimer = setTimeout(pollJob, 1200);
  } catch (error) { showStep(1); toast(error.message); }
}
function renderResults() {
  const candidates = state.result.candidates || []; showStep(3); $('#result-summary').textContent = `${escapeHtml(state.result.filename || state.file?.name)} から ${candidates.length}件の候補を見つけました。`;
  $('#result-body').innerHTML = candidates.map(item => `<tr><td>${formatClock(item.start_seconds)}</td><td>${formatClock(item.end_seconds)}</td><td>${escapeHtml(item.resume_text || '（発言なし）')}</td></tr>`).join('');
  $('#empty-result').classList.toggle('hidden', candidates.length > 0); $('#copy-button').disabled = candidates.length === 0;
  const warnings = state.result.warnings || []; $('#warning-banner').classList.toggle('hidden', !warnings.length); $('#warning-banner').textContent = warnings.join(' ');
}
async function copyResults() {
  const rows = [['開始','終了','カット終了後の話し始め'], ...(state.result?.candidates || []).map(item => [formatClock(item.start_seconds),formatClock(item.end_seconds),item.resume_text || ''])];
  const text = rows.map(row => row.join('\t')).join('\n'); try { await navigator.clipboard.writeText(text); } catch { const area=document.createElement('textarea');area.value=text;document.body.append(area);area.select();document.execCommand('copy');area.remove(); } toast('スプレッドシート用にコピーしました');
}

$('#drop-zone').addEventListener('click', () => $('#video-input').click()); $('#video-input').addEventListener('change', event => chooseFile(event.target.files[0]));
['dragenter','dragover'].forEach(type => $('#drop-zone').addEventListener(type,event => { event.preventDefault(); $('#drop-zone').classList.add('dragover'); }));
['dragleave','drop'].forEach(type => $('#drop-zone').addEventListener(type,event => { event.preventDefault(); $('#drop-zone').classList.remove('dragover'); })); $('#drop-zone').addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
$('#remove-file').addEventListener('click', clearFile); $('#start-button').addEventListener('click', startAnalysis); $('#copy-button').addEventListener('click', copyResults);
$('#new-video-button').addEventListener('click', () => { clearFile(); state.result=null; state.jobId=null; showStep(1); });
$('#environment-button').addEventListener('click', () => checkEnvironment(true)); $('#close-dialog').addEventListener('click', () => $('#environment-dialog').close());
checkEnvironment(false);
