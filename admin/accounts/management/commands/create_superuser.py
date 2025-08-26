from django.core.management.base import BaseCommand
from accounts.models import User


class Command(BaseCommand):
    help = '管理者ユーザーを作成します'

    def add_arguments(self, parser):
        parser.add_argument('--username', type=str, default='admin', help='管理者のユーザーネーム')
        parser.add_argument('--email', type=str, default='admin@example.com', help='管理者のメールアドレス')
        parser.add_argument('--password', type=str, default='admin123456', help='管理者のパスワード')
        parser.add_argument('--first-name', type=str, default='Admin', help='名前')
        parser.add_argument('--last-name', type=str, default='User', help='姓')

    def handle(self, *args, **options):
        username = options['username']
        email = options['email']
        password = options['password']
        first_name = options['first_name']
        last_name = options['last_name']

        # 管理者ユーザー作成
        admin_user, created = User.objects.get_or_create(
            username=username,
            defaults={
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'role': User.Role.ADMIN,
                'is_active': True,
            }
        )

        if created:
            admin_user.set_password(password)
            admin_user.save()
            self.stdout.write(
                self.style.SUCCESS(
                    f'✅ 管理者ユーザーが作成されました！\n'
                    f'Username: {username}\n'
                    f'Email: {email}\n'
                    f'Password: {password}'
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠️ 管理者ユーザー（{username}）は既に存在します。')
            )