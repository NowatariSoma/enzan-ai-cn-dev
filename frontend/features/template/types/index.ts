// Template feature types
export interface TemplateConfig {
  title: string;
  description?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '7xl' | 'full';
}

export interface TemplateCardProps {
  title: string;
  description: string;
  index: number;
} 