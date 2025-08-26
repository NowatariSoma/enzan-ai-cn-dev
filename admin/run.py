#!/usr/bin/env python
"""
Custom Django runserver command to default to port 8080
"""
import os
import sys
from django.core.management import execute_from_command_line

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    
    # Add default port if runserver command without port specified
    if len(sys.argv) == 2 and sys.argv[1] == 'runserver':
        sys.argv.append('8080')
    
    execute_from_command_line(sys.argv)