# ベースイメージとしてNode.js 18を使用
FROM node:18-alpine AS base

# 依存関係をインストールするステージ
FROM base AS deps
# libc6-compatを追加してネイティブ依存関係の互換性を確保
RUN apk add --no-cache libc6-compat
WORKDIR /app

# package.jsonとpackage-lock.jsonをコピー
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# ビルドステージ
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# 環境変数ファイルをコピー（.env.sampleを.envとしてコピー）
COPY .env.sample .env

# Next.jsアプリケーションをビルド
RUN npm run build

# 本番実行ステージ
FROM base AS runner
WORKDIR /app

# 本番環境の設定
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# nextjsユーザーを作成
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# standaloneビルドの出力をコピー
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# publicディレクトリの内容をコピー
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

# 環境変数ファイルを実行環境にもコピー（ビルドステージから.envをコピー）
COPY --from=builder --chown=nextjs:nodejs /app/.env ./

# nextjsユーザーに所有権を変更
RUN chown -R nextjs:nodejs /app
USER nextjs

# ポート3000を公開
EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# アプリケーションを起動
CMD ["node", "server.js"] 