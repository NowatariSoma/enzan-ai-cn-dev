/** @type {import('next').NextConfig} */
const nextConfig = {
  // Dockerでのスタンドアロンビルドを有効化
  output: 'standalone',
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
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
          : 'http://auth:8000/:path*',
      },
      {
        source: '/api/v1/:path*',
        destination: isDev 
          ? 'http://localhost:8000/api/v1/:path*'
          : 'http://ai_ameasure:8000/api/v1/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
