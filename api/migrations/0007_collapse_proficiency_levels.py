from django.db import migrations

LEVEL_MAP = {
    'upper_intermediate': 'intermediate',
    'advanced': 'expert',
    'fluent': 'expert',
}

def collapse_levels(apps, schema_editor):
    Script = apps.get_model('api', 'Script')
    UserProfile = apps.get_model('api', 'UserProfile')

    for old, new in LEVEL_MAP.items():
        Script.objects.filter(level=old).update(level=new)
        UserProfile.objects.filter(proficiency_level=old).update(proficiency_level=new)

def reverse_noop(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_userscriptprogress'),
    ]

    operations = [
        migrations.RunPython(collapse_levels, reverse_noop),
    ]