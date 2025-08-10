'use client';

import React from 'react';

export default function StructuralDiagram() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-white">
      <svg
        width="400"
        height="300"
        viewBox="0 0 400 300"
        className="w-full h-full max-w-md"
      >
        {/* Background grid lines */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect width="400" height="300" fill="url(#grid)" />
        
        {/* Ground line */}
        <line x1="50" y1="200" x2="350" y2="200" stroke="#333" strokeWidth="2" />
        
        {/* Foundation structure */}
        <rect x="80" y="200" width="240" height="60" fill="none" stroke="#333" strokeWidth="2" />
        
        {/* Arch/dome structure */}
        <path
          d="M 80 200 Q 200 120 320 200"
          fill="none"
          stroke="#3B82F6"
          strokeWidth="3"
        />
        
        {/* Support points */}
        <circle cx="80" cy="200" r="4" fill="#3B82F6" />
        <circle cx="200" cy="140" r="4" fill="#3B82F6" />
        <circle cx="320" cy="200" r="4" fill="#3B82F6" />
        
        {/* Displacement arrows and labels */}
        {/* Center displacement */}
        <line x1="200" y1="140" x2="200" y2="100" stroke="#E91E63" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="190" y1="110" x2="210" y2="130" stroke="#E91E63" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Left settlement */}
        <line x1="80" y1="200" x2="80" y2="240" stroke="#E91E63" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="70" y1="230" x2="90" y2="210" stroke="#E91E63" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Right settlement */}
        <line x1="320" y1="200" x2="320" y2="240" stroke="#E91E63" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="310" y1="230" x2="330" y2="210" stroke="#E91E63" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Reference line */}
        <line x1="30" y1="50" x2="150" y2="80" stroke="#333" strokeWidth="2" />
        
        {/* Labels */}
        {/* Displacement label */}
        <ellipse cx="280" cy="60" rx="35" ry="15" fill="none" stroke="#3B82F6" strokeWidth="2" />
        <text x="280" y="65" textAnchor="middle" className="text-sm font-medium" fill="#333">
          変位量
        </text>
        
        {/* Settlement label */}
        <ellipse cx="360" cy="180" rx="25" ry="15" fill="none" stroke="#E91E63" strokeWidth="2" />
        <text x="360" y="185" textAnchor="middle" className="text-sm font-medium" fill="#333">
          沈下量
        </text>
        
        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill="#E91E63"
            />
          </marker>
        </defs>
      </svg>
    </div>
  );
}