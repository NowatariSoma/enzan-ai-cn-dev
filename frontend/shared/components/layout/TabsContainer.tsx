'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/navigation/tabs';
import { TabsContainerProps } from '@/shared/types/tabs';

export function TabsContainer({
  tabs,
  activeTab,
  onTabChange,
  className = '',
  tabsListClassName = '',
  tabsContentClassName = ''
}: TabsContainerProps) {
  return (
    <Tabs value={activeTab} onValueChange={onTabChange} className={className}>
      <TabsList className={`grid w-full ${tabsListClassName}`} style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)` }}>
        {tabs.map((tab) => (
          <TabsTrigger key={tab.value} value={tab.value}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {tabs.map((tab) => (
        <TabsContent 
          key={tab.value} 
          value={tab.value} 
          className={tabsContentClassName}
        >
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  );
} 