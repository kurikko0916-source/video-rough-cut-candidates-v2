const $ = selector => document.querySelector(selector);
const state = { file: null, videoUrl: null, jobId: null, result: null, pollTimer: null, cloudMode: false, bridgeMode: false };
const views = { 1: $('#select-view'), 2: $('#analyze-view'), 3: $('#result-view'), 4: $('#rough-analyze-view'), 5: $('#rough-result-view') };
const phaseProgress = { queued: 5, upload: 12, probing: 20, audio: 36, transcribe: 62, format: 92, complete: 100 };
const phaseMessages = {
  queued: '処理の準備をしています…', upload: '動画をMac内に読み込みました', probing: '動画情報を確認しています…',
  audio: '文字起こし用の音声を取り出しています…', transcribe: '日本語の会話を文字起こししています…',
  format: 'タイムコード付きの表に整えています…', complete: '文字起こしが完了しました'
};

function toast(message) { const el = $('#toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove('show'), 2200); }
function showStep(view, step = view) {
  Object.entries(views).forEach(([key, element]) => element.classList.toggle('active-view', Number(key) === view));
  document.querySelectorAll('.step').forEach(el => { const value = Number(el.dataset.step); el.classList.toggle('active', value === step); el.classList.toggle('done', value < step); });
}
function formatBytes(bytes) { if (!bytes) return '0 MB'; const mb = bytes / 1024 / 1024; return `${mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb.toFixed(1) + ' MB'}`; }
function formatClock(seconds) { const safe = Math.max(0, Number(seconds) || 0); const h = Math.floor(safe / 3600); const m = Math.floor((safe % 3600) / 60); const s = Math.floor(safe % 60); return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }
function formatPrecise(seconds) { const centiseconds = Math.round(Math.max(0, Number(seconds) || 0) * 100); return `${formatClock(Math.floor(centiseconds / 100))}.${String(centiseconds % 100).padStart(2,'0')}`; }
function formatSrtTime(seconds) {
  const totalMilliseconds = Math.round(Math.max(0, Number(seconds) || 0) * 1000);
  const milliseconds = totalMilliseconds % 1000;
  const totalSeconds = Math.floor(totalMilliseconds / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  return `${String(hours).padStart(2,'0')}:${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')},${String(milliseconds).padStart(3,'0')}`;
}
function formatElapsed(seconds) { const value = Math.round(Number(seconds) || 0); return value >= 60 ? `${Math.floor(value / 60)}分${value % 60}秒` : `${value}秒`; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

function chooseFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mp4') && file.type !== 'video/mp4') return toast('MP4動画を選んでください');
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(file);
  state.file = file; $('#file-name').textContent = file.name; $('#file-meta').textContent = `${formatBytes(file.size)} ・ MP4`;
  $('#file-card').classList.remove('hidden'); $('#start-button').disabled = false;
  const video = document.createElement('video'); video.preload = 'metadata'; video.onloadedmetadata = () => { $('#file-meta').textContent = `${formatClock(video.duration)} ・ ${formatBytes(file.size)} ・ MP4`; }; video.src = state.videoUrl;
}
function clearFile() { if (state.videoUrl) URL.revokeObjectURL(state.videoUrl); state.file = null; state.videoUrl = null; $('#video-input').value = ''; $('#file-card').classList.add('hidden'); $('#start-button').disabled = true; $('#result-video').removeAttribute('src'); }

async function checkEnvironment(showDialog = false) {
  try {
    const response = await fetch('/api/health'); const data = await response.json(); const tools = data.tools || {};
    state.cloudMode = Boolean(data.cloud_mode);
    state.bridgeMode = Boolean(data.bridge_mode);
    const requiredReady = state.cloudMode || (state.bridgeMode ? tools.ffmpeg?.ready : (tools.ffmpeg?.ready && tools.ffprobe?.ready && tools.whisper?.ready && tools.whisper_model?.ready));
    const privacy = $('#privacy-description');
    if (privacy && state.bridgeMode) privacy.textContent = '映像はMac内で音声に変換し、軽い音声データだけをクラウドへ送ります。';
    else if (privacy && state.cloudMode) privacy.textContent = '選んだ動画を暗号化通信でクラウドへ送り、処理後に削除します。';
    $('#environment-dot').className = `status-dot ${requiredReady ? 'ready' : 'error'}`;
    if (showDialog) {
      const labels = { ffmpeg:'FFmpeg（音声抽出）', ffprobe:'ffprobe（動画情報）', whisper:'whisper.cpp（文字起こし）', whisper_model:'Whisperモデル' };
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
    let response;
    if (state.cloudMode) {
      const sessionResponse = await fetch('/api/uploads', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:state.file.name,size:state.file.size})});
      const session = await sessionResponse.json(); if (!sessionResponse.ok) throw new Error(session.error || 'アップロードを準備できませんでした');
      setProgress('upload', 7);
      const uploadResponse = await fetch(session.upload_url, {method:'PUT', headers:{'Content-Type':'video/mp4'}, body:state.file});
      if (!uploadResponse.ok) throw new Error('動画をCloud Storageへアップロードできませんでした');
      response = await fetch('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_id:session.job_id,object_name:session.object_name})});
    } else {
      response = await fetch('/api/jobs', { method:'POST', headers:{'Content-Type':'video/mp4','X-Filename':encodeURIComponent(state.file.name),'X-File-Size':String(state.file.size)}, body:state.file });
    }
    const data = await response.json(); if (!response.ok) throw new Error(data.error || '動画を読み込めませんでした');
    state.jobId = data.job_id; localStorage.setItem('roughCutLastJobId', state.jobId); pollJob();
  } catch (error) { showStep(1); toast(error.message); }
}
function setProgress(phase, serverProgress) {
  const progress = Number(serverProgress ?? phaseProgress[phase] ?? 0); $('#progress-bar').style.width = `${progress}%`; $('#progress-percent').textContent = `${progress}%`; $('#progress-message').textContent = phaseMessages[phase] || '処理中です…';
  const order = ['upload','audio','transcribe']; const current = Math.max(0, order.indexOf(phase === 'probing' ? 'upload' : phase === 'format' ? 'transcribe' : phase));
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
  const segments = state.result.segments || []; showStep(3); $('#result-summary').textContent = `${state.result.filename || state.file?.name} の全体を ${segments.length}個の発言として文字起こししました。`;
  $('#result-video').src = state.videoUrl || '';
  $('#meta-duration').textContent = formatClock(state.result.duration_seconds); $('#meta-segments').textContent = `${segments.length}件`; $('#meta-processing').textContent = formatElapsed(state.result.processing_seconds); $('#meta-model').textContent = state.result.transcription_model || '—';
  $('#result-body').innerHTML = segments.map((item,index) => `<tr><td><button class="time-button" type="button" data-segment-index="${index}">${formatPrecise(item.start)}</button></td><td>${formatPrecise(item.end)}</td><td>${escapeHtml(item.text)}</td></tr>`).join('');
  $('#empty-result').classList.toggle('hidden', segments.length > 0); $('#copy-button').disabled = segments.length === 0;
  const warnings = state.result.warnings || []; $('#warning-banner').classList.toggle('hidden', !warnings.length); $('#warning-banner').textContent = warnings.join(' ');
}

async function startRoughCuts() {
  if (!state.jobId) return toast('文字起こし結果を確認できません');
  showStep(4, 4); setRoughProgress('queued', 5);
  try {
    const response = await fetch(`/api/jobs/${state.jobId}/rough-cuts`, {method:'POST'});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'AI判定を開始できません');
    pollRoughCuts();
  } catch (error) { showStep(3, 3); toast(error.message); }
}
function setRoughProgress(phase, progress) {
  const messages = {queued:'ローカルAIを準備しています…',understanding:'動画全体のテーマを読み取っています…',evaluating:'大きく削れそうなまとまりを評価しています…',safety:'必要な本編を削っていないか確認しています…'};
  const value = Number(progress || 0); $('#rough-progress-bar').style.width = `${value}%`; $('#rough-progress-percent').textContent = `${value}%`; $('#rough-progress-message').textContent = messages[phase] || '分析中です…';
  const index = {queued:0,understanding:0,evaluating:1,safety:2,complete:2}[phase] ?? 0;
  $('#rough-analyze-view').querySelectorAll('.process-list span').forEach((el,i) => { el.classList.toggle('active', i === index); el.classList.toggle('done', i < index || phase === 'complete'); });
}
async function pollRoughCuts() {
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`); const data = await response.json(); if (!response.ok) throw new Error(data.error || '状態を取得できません');
    state.result = data; setRoughProgress(data.rough_phase, data.rough_progress);
    if (data.rough_status === 'complete') return renderRoughCuts();
    if (data.rough_status === 'failed') throw new Error(data.rough_error || 'AI判定に失敗しました');
    state.pollTimer = setTimeout(pollRoughCuts, 1400);
  } catch (error) { showStep(3, 3); toast(`${error.message}（文字起こしは保存済みです）`); }
}
function renderList(selector, values) { $(selector).innerHTML = (values || []).map(value => `<li>${escapeHtml(value)}</li>`).join(''); }
function renderRoughCuts() {
  const candidates = state.result.candidates || [], understanding = state.result.video_understanding || {};
  showStep(5, 4); $('#central-theme').textContent = understanding.central_theme || '—'; renderList('#main-conclusions', understanding.main_conclusions); renderList('#important-points', understanding.important_points);
  const strongCount = candidates.filter(item => (item.decision || 'strong') === 'strong').length;
  const reviewCount = candidates.filter(item => item.decision === 'review').length;
  $('#rough-summary').textContent = `強いカット候補 ${strongCount}件、カット検討候補 ${reviewCount}件を見つけました。`;
  $('#candidate-body').innerHTML = candidates.map((item,index) => { const decision = item.decision || 'strong'; const label = decision === 'review' ? 'カット検討' : '強い候補'; return `<tr><td><span class="decision-badge ${decision}">${label}</span></td><td><button class="time-button" type="button" data-candidate-index="${index}">${formatPrecise(item.start_seconds)}</button></td><td>${formatPrecise(item.end_seconds)}</td><td>${escapeHtml(item.resume_text || '—')}</td><td><strong>${escapeHtml(item.category)}</strong><br>${escapeHtml(item.reason)}</td></tr>`; }).join('');
  $('#empty-candidates').classList.toggle('hidden', candidates.length > 0); $('#copy-candidates-button').disabled = candidates.length === 0;
}
async function copyCandidates() {
  const rows = [['判定','開始','終了','カット終了後の話し始め'], ...(state.result?.candidates || []).map(item => [(item.decision || 'strong') === 'review' ? 'カット検討候補' : '強いカット候補',formatClock(item.start_seconds),formatClock(item.end_seconds),item.resume_text || ''])];
  const value = rows.map(row => row.join('\t')).join('\n'); try { await navigator.clipboard.writeText(value); } catch { const area=document.createElement('textarea');area.value=value;document.body.append(area);area.select();document.execCommand('copy');area.remove(); } toast('カット指示をコピーしました');
}
async function copyResults() {
  const rows = [['開始','終了','発言内容'], ...(state.result?.segments || []).map(item => [formatPrecise(item.start),formatPrecise(item.end),item.text || ''])];
  const text = rows.map(row => row.join('\t')).join('\n'); try { await navigator.clipboard.writeText(text); } catch { const area=document.createElement('textarea');area.value=text;document.body.append(area);area.select();document.execCommand('copy');area.remove(); } toast('タイムコード付き文字起こしをコピーしました');
}
function downloadSrt() {
  const segments = (state.result?.segments || []).filter(item => String(item.text || '').trim());
  if (!segments.length) return toast('保存できる文字起こしがありません');
  const srt = segments.map((item, index) => `${index + 1}\n${formatSrtTime(item.start)} --> ${formatSrtTime(item.end)}\n${String(item.text).trim()}`).join('\n\n') + '\n';
  const sourceName = state.result?.filename || state.file?.name || '文字起こし';
  const baseName = sourceName.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]/g, '_');
  const blob = new Blob(['\uFEFF', srt], {type:'application/x-subrip;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = `${baseName}_Vrew用.srt`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('Vrew用SRTを保存しました');
}

$('#drop-zone').addEventListener('click', () => $('#video-input').click()); $('#video-input').addEventListener('change', event => chooseFile(event.target.files[0]));
['dragenter','dragover'].forEach(type => $('#drop-zone').addEventListener(type,event => { event.preventDefault(); $('#drop-zone').classList.add('dragover'); }));
['dragleave','drop'].forEach(type => $('#drop-zone').addEventListener(type,event => { event.preventDefault(); $('#drop-zone').classList.remove('dragover'); })); $('#drop-zone').addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
$('#remove-file').addEventListener('click', clearFile); $('#start-button').addEventListener('click', startAnalysis); $('#copy-button').addEventListener('click', copyResults); $('#download-srt-button').addEventListener('click', downloadSrt); $('#rough-download-srt-button').addEventListener('click', downloadSrt);
$('#rough-cut-button').addEventListener('click', startRoughCuts); $('#copy-candidates-button').addEventListener('click', copyCandidates); $('#back-transcript-button').addEventListener('click', () => { renderResults(); showStep(3,3); });
$('#candidate-body').addEventListener('click', event => { const button=event.target.closest('[data-candidate-index]'); if(!button)return; const item=state.result?.candidates?.[Number(button.dataset.candidateIndex)]; const video=$('#result-video'); if(item&&video.src){video.currentTime=Math.max(0,Number(item.start_seconds)-1);video.play().catch(()=>{});} });
$('#result-body').addEventListener('click', event => { const button = event.target.closest('[data-segment-index]'); if (!button) return; const segment = state.result?.segments?.[Number(button.dataset.segmentIndex)]; if (!segment) return; const video = $('#result-video'); video.currentTime = Math.max(0, Number(segment.start) - 1); video.play().catch(() => {}); });
$('#new-video-button').addEventListener('click', () => { clearFile(); state.result=null; state.jobId=null; showStep(1); });
$('#new-video-from-rough-button').addEventListener('click', () => { clearFile(); state.result=null; state.jobId=null; showStep(1); });
$('#environment-button').addEventListener('click', () => checkEnvironment(true)); $('#close-dialog').addEventListener('click', () => $('#environment-dialog').close());
async function restoreSavedJob() {
  // 通常表示では前回の大きな文字起こし結果を取得しない。
  // 結果を共有・再表示する明示的な ?job=... URLのときだけ読み込む。
  const requested = new URLSearchParams(location.search).get('job');
  if (!requested || !/^[0-9a-f]+$/.test(requested)) return;
  try {
    const response = await fetch(`/api/jobs/${requested}`); if (!response.ok) return;
    const data = await response.json(); if (data.status !== 'complete') return;
    state.jobId=requested; state.result=data; localStorage.setItem('roughCutLastJobId', requested);
    if (data.rough_status === 'complete') renderRoughCuts(); else renderResults();
  } catch {}
}
checkEnvironment(false); restoreSavedJob();
