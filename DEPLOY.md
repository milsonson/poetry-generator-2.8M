# 公网真实生成部署

GitHub Pages 只能托管静态网页，不能运行 `web_app.py`、PyTorch 或 `transformer_poetry.pth`。要让公网网页和本地完全一样实时生成，需要部署一个 Python 后端。

本仓库已经包含 Docker 部署入口，容器启动后会运行：

```bash
python web_app.py
```

服务会加载当前仓库里的：

- `transformer_poetry.pth`
- `vocab.json`
- `static/index.html`

## Hugging Face Spaces

1. 打开 Hugging Face，创建一个新的 Space。
2. SDK 选择 `Docker`。
3. 把这个 GitHub 仓库导入或把仓库内容推到 Space。
4. Space 构建完成后，打开 Space URL。

这个 URL 才是真实生成版本。页面里的“生成”按钮会请求 `/api/generate`，后端会调用同一个 checkpoint 实时采样。

## 本地 Docker 验证

```bash
docker build -t poetry-generator-2-8m .
docker run --rm -p 7860:7860 poetry-generator-2-8m
```

然后打开：

```text
http://127.0.0.1:7860
```

