# core/urls.py
from django.contrib import admin
from django.urls    import path, include
from django.conf    import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/',    admin.site.urls),
    path('api/',      include('api.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Full URL list after setup:
# POST  /api/register/   <- create new account
# POST  /api/login/      <- get auth token
# POST  /api/analyze/    <- submit audio for analysis  (main endpoint)
# GET   /api/sessions/   <- get user session history
# GET   /api/profile/    <- get/update user profile
# PATCH /api/profile/
# GET   /admin/          <- Django admin panel