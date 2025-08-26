from rest_framework import filters


class CustomOrderingFilter(filters.OrderingFilter):
    ordering_param = 'sort_by'
    ordering_description = 'Which field to use when ordering the results.'

    def get_ordering(self, request, queryset, view):
        ordering = super().get_ordering(request, queryset, view)
        if not ordering:
            return ordering

        # Get sort_order parameter (default to 'desc' as per spec)
        sort_order = request.query_params.get('sort_order', 'desc')

        # Apply sort order to all fields
        if sort_order.lower() == 'asc':
            return [field.lstrip('-') for field in ordering]
        return [f'-{field}' if not field.startswith('-') else field for field in ordering]
