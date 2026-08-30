# utils/permissions.py
from kivy.utils import platform

def request_android_permissions():
    """
    Requests necessary permissions based on Android API version.
    Handles the shift from READ_EXTERNAL_STORAGE to READ_MEDIA_* in API 33+.
    """
    if platform != "android":
        return

    from android.permissions import request_permissions, Permission
    from android import api_version

    # Common permissions
    permissions = [Permission.CAMERA, Permission.RECORD_AUDIO]

    # Android 13+ (API 33) requires granular media permissions
    if api_version >= 33:
        permissions.append(Permission.READ_MEDIA_IMAGES)
        permissions.append(Permission.READ_MEDIA_VIDEO)
        # WRITE_EXTERNAL_STORAGE is deprecated/no-op for media in API 33+, 
        # but Kivy's file chooser might still check it.
    else:
        # Android 12 and below
        permissions.append(Permission.READ_EXTERNAL_STORAGE)
        permissions.append(Permission.WRITE_EXTERNAL_STORAGE)

    request_permissions(permissions)