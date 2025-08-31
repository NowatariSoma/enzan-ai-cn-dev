/** @type {import('next').NextConfig} */
const nextConfig = {
  // Dockerでのスタンドアロンビルドを有効化
  output: 'standalone',
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
  // Remove experimental.skipMiddlewareUrlNormalize (deprecated)
  images: {
    unoptimized: true,
  },
  webpack: (config, { dev, isServer }) => {
    // Disable CSS optimization in production to avoid build errors
    if (!dev && !isServer) {
      config.optimization.minimize = false;
      config.optimization.minimizer = [];
    }
    return config;
  },
  async rewrites() {
    const isDev = process.env.NODE_ENV === 'development';
    return [
      {
        source: '/api/db/:path*',
        destination: isDev 
          ? 'http://localhost:8001/:path*'
          : 'http://admin:8080/:path*',
      },
      {
        source: '/api/v1/:path*',
        destination: 'http://mlp_ai_ameasure:8000/api/v1/:path*',
      },
    ];
  },
  // Long timeout for heavy API operations
  experimental: {
    proxyTimeout: 600000, // 10 minutes timeout
  },
  // HTTP keep-alive for better connection stability
  serverRuntimeConfig: {
    httpAgentOptions: {
      keepAlive: true,
      keepAliveMsecs: 30000, // 30 seconds
    },
  },
};

module.exports = nextConfig;
