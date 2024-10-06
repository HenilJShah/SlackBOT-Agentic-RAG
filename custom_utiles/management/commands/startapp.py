from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
import os


class Command(BaseCommand):
    help = "Custom startapp command that automatically creates a urls.py file"

    def add_arguments(self, parser):
        parser.add_argument("name", type=str, help="Name of the application")

    def handle(self, *args, **options):
        app_name = options["name"]

        # Call the original startapp command to create the app structure
        call_command("startapp", app_name)

        # Define the path for the new urls.py file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_dir = os.path.join(project_root, app_name)
        urls_file_path = os.path.join(app_dir, "urls.py")

        # Check if the app was successfully created and create the urls.py file
        if os.path.exists(app_dir):
            # Create the urls.py file with some boilerplate code
            with open(urls_file_path, "w") as urls_file:
                urls_file.write(
                    "from django.urls import path\n\n"
                    "urlpatterns = [\n"
                    "    # Add your URL patterns here\n"
                    "]\n"
                )
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully created urls.py for app "{app_name}".'
                )
            )
        else:
            raise CommandError(f'Failed to create the app "{app_name}".')
