'use client';

import React from 'react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { Button } from '@/components/ui/forms/button';
import { Badge } from '@/components/ui/data-display/badge';
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
  Brain,
  MapPin,
  Star,
  Search,
  AlertTriangle,
  Monitor
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Image from 'next/image';
import { locations, getLocationsByRegion } from '@/lib/data/locations';
import { useFavoriteLocations } from '@/lib/hooks/useFavoriteLocations';

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
  const [expandedSections, setExpandedSections] = useState<string[]>(['favorites', 'locations', 'ai-measure']);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
  const pathname = usePathname();
  const router = useRouter();
  const { favoriteIds, toggleFavorite, isFavorite } = useFavoriteLocations();

  // 初回読み込み時に選択された拠点を復元
  useEffect(() => {
    const stored = localStorage.getItem('selectedLocation');
    if (stored) {
      try {
        const location = JSON.parse(stored);
        if (location && location.id) {
          setSelectedLocation(location.id);
        }
      } catch (error) {
        console.error('Failed to parse selected location:', error);
      }
    }
  }, []);

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };

  const handleNavigateAndClose = (path: string) => {
    router.push(path);
    if (onMobileClose) {
      onMobileClose();
    }
  };

  // お気に入り拠点を取得
  const favoriteLocations = locations.filter(loc => favoriteIds.includes(loc.id));
  
  // 検索フィルタリング
  const filteredLocations = locations.filter(loc => 
    loc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    loc.tunnelName.toLowerCase().includes(searchQuery.toLowerCase()) ||
    loc.prefecture.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleLocationClick = (locationId: string) => {
    const location = locations.find(loc => loc.id === locationId);
    if (location) {
      localStorage.setItem('selectedLocation', JSON.stringify(location));
      setSelectedLocation(locationId);
      // 拠点専用ダッシュボードへ遷移
      handleNavigateAndClose(`/location/${locationId}`);
    }
  };

  // 選択中の拠点を取得
  const currentLocation = selectedLocation ? locations.find(loc => loc.id === selectedLocation) : null;

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

          {/* 拠点管理セクション */}
          {!isCollapsed && (
            <>
              {/* 検索ボックス */}
              <div className="px-2 py-2">
                <div className="relative">
                  <Search className="absolute left-2 top-2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="拠点を検索..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-8 pr-2 py-1.5 text-sm border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              {/* お気に入り拠点 */}
              <NavItem
                icon={<Star className="w-4 h-4" />}
                label="お気に入り拠点"
                hasChildren={true}
                isExpanded={expandedSections.includes('favorites')}
                onToggleExpand={() => toggleSection('favorites')}
              />
              
              {expandedSections.includes('favorites') && (
                <div className="ml-4 space-y-1">
                  {favoriteLocations.length === 0 ? (
                    <div className="px-3 py-2 text-xs text-gray-500">
                      お気に入りなし
                    </div>
                  ) : (
                    favoriteLocations.map(location => (
                      <div key={location.id} className="flex items-center justify-between pr-2 group">
                        <button
                          onClick={() => handleLocationClick(location.id)}
                          className="flex-1 flex items-center px-2 py-1.5 text-sm text-gray-700 rounded hover:bg-gray-100"
                        >
                          <MapPin className="w-3 h-3 mr-2 text-gray-400" />
                          <span className="truncate">{location.tunnelName}</span>
                          {location.alertLevel === 'danger' && (
                            <AlertTriangle className="w-3 h-3 ml-1 text-rose-500" />
                          )}
                        </button>
                        <button
                          onClick={() => toggleFavorite(location.id)}
                          className="p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Star className="w-3 h-3 text-amber-500 fill-amber-500" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* 全拠点 */}
              <NavItem
                icon={<MapPin className="w-4 h-4" />}
                label="全拠点"
                hasChildren={true}
                isExpanded={expandedSections.includes('locations')}
                onToggleExpand={() => toggleSection('locations')}
              />
              
              {expandedSections.includes('locations') && (
                <div className="ml-4 space-y-1 max-h-64 overflow-y-auto">
                  {(searchQuery ? filteredLocations : locations).map(location => (
                    <div key={location.id} className="flex items-center justify-between pr-2 group">
                      <button
                        onClick={() => handleLocationClick(location.id)}
                        className="flex-1 flex items-center px-2 py-1.5 text-sm text-gray-700 rounded hover:bg-gray-100"
                      >
                        <MapPin className="w-3 h-3 mr-2 text-gray-400" />
                        <span className="truncate">{location.tunnelName}</span>
                        {location.alertLevel === 'danger' && (
                          <AlertTriangle className="w-3 h-3 ml-1 text-rose-500" />
                        )}
                        {location.status === 'active' && (
                          <div className="w-2 h-2 ml-1 bg-green-500 rounded-full animate-pulse" />
                        )}
                      </button>
                      <button
                        onClick={() => toggleFavorite(location.id)}
                        className="p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Star className={cn(
                          "w-3 h-3",
                          isFavorite(location.id) ? "text-amber-500 fill-amber-500" : "text-gray-400"
                        )} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

            </>
          )}

          {/* 拠点個別機能セクション - 拠点に基づいて動的に表示 */}
          {!isCollapsed && currentLocation?.availableFeatures?.aiMeasurement && (
            <>
              <NavTitle label="拠点個別機能" />
              
              {/* 選択中の拠点表示 */}
              {currentLocation && (
                <div 
                  className="mx-2 mb-2 p-2 bg-slate-50 border border-slate-200 rounded-md cursor-pointer hover:bg-slate-100 transition-colors"
                  onClick={() => handleLocationClick(currentLocation.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <MapPin className="w-3 h-3 mr-1 text-blue-600" />
                      <span className="text-xs font-medium text-blue-900 truncate">
                        {currentLocation.tunnelName}
                      </span>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedLocation(null);
                      }}
                      className="text-blue-600 hover:text-blue-800"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              )}

              <NavItem
                icon={<Activity className="w-4 h-4" />}
                label="AI-A計測集計"
                hasChildren={true}
                isExpanded={expandedSections.includes('ai-measure')}
                onToggleExpand={() => toggleSection('ai-measure')}
              />
              
              {expandedSections.includes('ai-measure') && (
                <div className="ml-4 space-y-1">
                  {currentLocation?.availableFeatures?.measurement && (
                    <SubNavItem
                      icon={<BarChart3 className="w-4 h-4" />}
                      label="A計測集計"
                      active={pathname === '/measurements'}
                      href={currentLocation ? `/measurements?location=${currentLocation.id}` : '/measurements'}
                    />
                  )}
                  
                  {currentLocation?.availableFeatures?.simulation && (
                    <SubNavItem
                      icon={<Activity className="w-4 h-4" />}
                      label="最終変位・沈下予測"
                      active={pathname === '/simulation'}
                      href={currentLocation ? `/simulation?location=${currentLocation.id}` : '/simulation'}
                    />
                  )}
                  
                  {currentLocation?.availableFeatures?.modelCreation && (
                    <SubNavItem
                      icon={<Brain className="w-4 h-4" />}
                      label="予測モデル作成"
                      active={pathname === '/learning'}
                      href={currentLocation ? `/learning?location=${currentLocation.id}` : '/learning'}
                    />
                  )}
                </div>
              )}
            </>
          )}
          
          {/* 拠点が選択されていない場合のメッセージ */}
          {!isCollapsed && !currentLocation && (
            <div className="mx-2 mb-2 p-3 bg-white border border-gray-200 rounded-md">
              <p className="text-xs text-gray-600 text-center">
                拠点を選択すると<br />利用可能な機能が表示されます
              </p>
            </div>
          )}

          {/* 追加機能セクション - 拠点に基づいて動的に表示 */}
          {!isCollapsed && currentLocation && (
            currentLocation.availableFeatures.realTimeMonitoring
          ) && (
            <>
              <NavTitle label="追加機能" />
              
              {currentLocation.availableFeatures.realTimeMonitoring && (
                <NavItem
                  icon={<Monitor className="w-4 h-4" />}
                  label="リアルタイム監視"
                  active={pathname === '/monitoring'}
                  href={`/monitoring?location=${currentLocation.id}`}
                />
              )}
            </>
          )}

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