import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // 認証チェックを無効化 - すべてのルートにアクセス可能
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - favicon.png (favicon file)
     * - images/ (image files)
     * - icons/ (icon files)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|favicon.png|images|icons).*)',
  ],
}; 