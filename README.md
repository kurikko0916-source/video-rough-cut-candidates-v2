# 粗カットAI

建築・住宅系YouTubeの日本語対談動画を解析するMac用ローカルアプリです。現在のSTEP1では、動画全体のタイムコード付き文字起こしと精度確認だけを行います。

## MVPの方針

- 動画・音声・文字起こしはMac内で処理
- 有料APIは使用しない
- STEP1ではCUT／KEEP判定を行わない
- 開始・終了・発言内容の3列を出力
- 発言のタイムコードから動画を再生できる
- STEP2で数十秒〜数分単位の粗カット判定を追加予定

## ローカルツール

- 専用Python 3.12
- FFmpeg / ffprobe
- whisper.cpp (`whisper-cli`)
- Ollama

これらはApple開発ツールを使わず、`tools/` のプロジェクト専用環境に配置します。Mac全体の開発環境は変更しません。

## 起動

`start.command` をダブルクリックします。Ollamaとアプリが自動起動し、ブラウザーで `http://127.0.0.1:8765` が開きます。

whisper.cppのモデルを `models/ggml-large-v3-turbo-q5_0.bin` に置くか、環境変数 `WHISPER_MODEL` で指定します。Ollamaのモデル名は `OLLAMA_MODEL` で変更できます（初期値: `qwen3:4b`）。

解析中の一時ファイルと結果は `data/jobs/` へ保存され、GitHubには送られません。

## 将来拡張の入れ物

`config/channel_rules.json` に、将来のチャンネル固有ルール・残す例・カット例を分離して追加できる構造を用意しています。MVPでは無効です。
