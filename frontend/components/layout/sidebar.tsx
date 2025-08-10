'use client';

import React from 'react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { Button } from '@/components/ui/forms/button';
import { 
  ChevronLeft, 
  ChevronRight, 
  FileText,
  Users,
  X,
  Home,
  Palette,
  Settings,
  ChevronUp,
  ChevronDown,
  Activity,
  BarChart3,
  Brain
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Image from 'next/image';

interface SidebarProps {
  isMobileOpen?: boolean;
  onMobileClose?: () => void;
}

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick?: () => void;
  href?: string;
  className?: string;
  hasChildren?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

function NavItem({ icon, label, active, onClick, href, className, hasChildren, isExpanded, onToggleExpand }: NavItemProps) {
  const handleChevronClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onToggleExpand) {
      onToggleExpand();
    }
  };

  const content = (
    <>
      <span className="flex-shrink-0">{icon}</span>
      <span className="ml-3 truncate flex-1 text-left">{label}</span>
      {hasChildren && (
        <div
          onClick={handleChevronClick}
          className="flex-shrink-0 ml-2 p-1 hover-icon transition-colors cursor-pointer"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleChevronClick(e as any);
            }
          }}
        >
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      )}
    </>
  );

  const baseClassName = cn(
    "flex items-center w-full px-3 py-2.5 text-sm font-medium rounded-md transition-all duration-200 hover-nav",
    active 
      ? "active-nav" 
      : "",
    className
  );

  if (href) {
    return (
      <Link href={href} className={baseClassName}>
        {content}
      </Link>
    );
  }

  return (
    <button onClick={onClick} className={baseClassName}>
      {content}
    </button>
  );
}

function SubNavItem({ icon, label, active, onClick, href, className }: NavItemProps) {
  const content = (
    <>
      <span className="flex-shrink-0">{icon}</span>
      <span className="ml-3 truncate">{label}</span>
    </>
  );

  const baseClassName = cn(
    "flex items-center w-full pl-10 pr-3 py-2 text-sm font-medium rounded-md transition-all duration-200 hover-nav",
    active 
      ? "active-nav" 
      : "",
    className
  );

  if (href) {
    return (
      <Link href={href} className={baseClassName}>
        {content}
      </Link>
    );
  }

  return (
    <button onClick={onClick} className={baseClassName}>
      {content}
    </button>
  );
}

function NavTitle({ label }: { label: string }) {
  return (
    <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
      {label}
    </div>
  );
}

export function Sidebar({ isMobileOpen, onMobileClose }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };


  const handleNavigateAndClose = (path: string) => {
    router.push(path);
    if (onMobileClose) {
      onMobileClose();
    }
  };

  // Mobile overlay
  if (isMobileOpen) {
    return (
      <>
        {/* Mobile backdrop */}
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={onMobileClose}
        />
        
        {/* Mobile sidebar */}
        <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-gray-200 shadow-lg md:hidden transform transition-transform duration-300 ease-in-out">
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 flex-shrink-0">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 flex items-center justify-center">
                  <Image 
                    src="/favicon.png" 
                    alt="演算工房" 
                    width={32} 
                    height={32}
                    className="rounded"
                  />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-gray-900">演算工房</h2>
                  <p className="text-xs text-gray-500">AI-CN</p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onMobileClose}
                className="hover-icon"
              >
                <X className="w-5 h-5" />
              </Button>
            </div>

            <div className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
              <NavTitle label="メイン" />
              
              <NavItem
                icon={<Home className="w-4 h-4" />}
                label="ダッシュボード"
                active={pathname === '/'}
                onClick={() => handleNavigateAndClose('/')}
              />

              <NavTitle label="AI-A計測" />

              <NavItem
                icon={<BarChart3 className="w-4 h-4" />}
                label="A計測集計"
                active={pathname === '/measurements'}
                onClick={() => handleNavigateAndClose('/measurements')}
              />

              <NavItem
                icon={<Brain className="w-4 h-4" />}
                label="予測モデル作成"
                active={pathname === '/learning'}
                onClick={() => handleNavigateAndClose('/learning')}
              />

              <NavItem
                icon={<Activity className="w-4 h-4" />}
                label="最終変位・沈下予測"
                active={pathname === '/simulation'}
                onClick={() => handleNavigateAndClose('/simulation')}
              />

              <NavTitle label="その他" />

              <NavItem
                icon={<Palette className="w-4 h-4" />}
                label="テンプレート"
                active={pathname === '/template'}
                onClick={() => handleNavigateAndClose('/template')}
              />

              <NavItem
                icon={<Users className="w-4 h-4" />}
                label="設定"
                active={pathname === '/settings'}
                onClick={() => handleNavigateAndClose('/settings')}
              />
            </div>
          </div>
        </div>
      </>
    );
  }

  // Desktop sidebar
  return (
    <div className={cn(
      "hidden md:flex md:flex-col md:fixed md:inset-y-0 md:left-0 transition-all duration-300 z-40",
      isCollapsed ? "w-16" : "w-64"
    )}>
      <div className="flex flex-col flex-1 min-h-0 bg-white border-r border-gray-200 shadow-sm">
        {/* Header */}
        <div className="flex items-center h-16 px-4 border-b border-gray-200 flex-shrink-0">
          {!isCollapsed ? (
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 flex items-center justify-center">
                <Image 
                  src="/favicon.png" 
                  alt="演算工房" 
                  width={32} 
                  height={32}
                  className="rounded"
                />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-gray-900">演算工房</h2>
                <p className="text-xs text-gray-500">AI-CN</p>
              </div>
            </div>
          ) : (
            <div className="w-8 h-8 flex items-center justify-center mx-auto">
              <Image 
                src="/favicon.png" 
                alt="演算工房" 
                width={32} 
                height={32}
                className="rounded"
              />
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
          {!isCollapsed && <NavTitle label="メイン" />}
          
          <NavItem
            icon={<Home className="w-4 h-4" />}
            label={isCollapsed ? "" : "ダッシュボード"}
            active={pathname === '/'}
            href="/"
            className={isCollapsed ? "justify-center px-2" : ""}
          />

          {!isCollapsed && <NavTitle label="AI-A計測" />}

          <NavItem
            icon={<BarChart3 className="w-4 h-4" />}
            label={isCollapsed ? "" : "A計測集計"}
            active={pathname === '/measurements'}
            href="/measurements"
            className={isCollapsed ? "justify-center px-2" : ""}
          />

          <NavItem
            icon={<Brain className="w-4 h-4" />}
            label={isCollapsed ? "" : "予測モデル作成"}
            active={pathname === '/learning'}
            href="/learning"
            className={isCollapsed ? "justify-center px-2" : ""}
          />

          <NavItem
            icon={<Activity className="w-4 h-4" />}
            label={isCollapsed ? "" : "最終変位・沈下予測"}
            active={pathname === '/simulation'}
            href="/simulation"
            className={isCollapsed ? "justify-center px-2" : ""}
          />

          {!isCollapsed && <NavTitle label="その他" />}

          <NavItem
            icon={<Palette className="w-4 h-4" />}
            label={isCollapsed ? "" : "テンプレート"}
            active={pathname === '/template'}
            href="/template"
            className={isCollapsed ? "justify-center px-2" : ""}
          />

          <NavItem
            icon={<Users className="w-4 h-4" />}
            label={isCollapsed ? "" : "設定"}
            active={pathname === '/settings'}
            href="/settings"
            className={isCollapsed ? "justify-center px-2" : ""}
          />
        </div>

        {/* Collapse toggle */}
        <div className="flex-shrink-0 border-t border-gray-200 p-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebar}
            className="w-full justify-center hover-icon"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </Button>
        </div>
      </div>
    </div>
  );
}

// Mobile toggle button component for header
export function MobileSidebarToggle({ onToggle }: { onToggle: () => void }) {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onToggle}
      className="md:hidden w-10 h-10 p-0 hover:bg-gray-100"
    >
      <FileText className="w-6 h-6 text-blue-600" />
    </Button>
  );
} 