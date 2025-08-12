from rest_framework.throttling import UserRateThrottle

class ProjectsRateThrottle(UserRateThrottle):
    """
    Rate limit of 1000 requests per hour per user as per specification.
    """
    rate = '1000/hour'
    scope = 'projects'