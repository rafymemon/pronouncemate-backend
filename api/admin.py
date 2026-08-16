# api/admin.py
from django.contrib import admin
from .models import PracticeSession, PhonemeError, UserProfile, UserScriptProgress

@admin.register(PracticeSession)
class SessionAdmin(admin.ModelAdmin):
    list_display  = ['user','accuracy','grade','total_words','flagged_words','created_at']
    list_filter   = ['grade','created_at']
    search_fields = ['user__username','script']
    readonly_fields = ['result_json','created_at']

@admin.register(PhonemeError)
class ErrorAdmin(admin.ModelAdmin):
    list_display  = ['session','word','error_type','expected','spoken','severity']
    list_filter   = ['error_type']
    search_fields = ['word']

@admin.register(UserProfile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ['user','proficiency_level','total_sessions','best_score']




@admin.register(UserScriptProgress)
class UserScriptProgressAdmin(admin.ModelAdmin):
    list_display = ('user', 'script', 'times_practiced', 'last_practiced_at', 'best_score', 'last_score')
    list_filter = ('script__mode', 'script__level')
    search_fields = ('user__username', 'script__title')