import { useState } from 'react';

export const useMobileSidebar = () => {
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const handleMobileSidebarToggle = () => {
    setIsMobileSidebarOpen(!isMobileSidebarOpen);
  };

  const handleMobileSidebarClose = () => {
    setIsMobileSidebarOpen(false);
  };

  return {
    isMobileSidebarOpen,
    handleMobileSidebarToggle,
    handleMobileSidebarClose,
  };
}; 