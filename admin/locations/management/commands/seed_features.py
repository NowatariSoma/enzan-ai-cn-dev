from django.core.management.base import BaseCommand
from locations.models import Feature


class Command(BaseCommand):
    help = '機能マスターのシードデータを作成します'

    def add_arguments(self, parser):
        parser.add_argument(
            '--delete',
            action='store_true',
            help='既存のデータを削除してから作成します',
        )

    def handle(self, *args, **options):
        if options['delete']:
            self.stdout.write('既存の機能データを削除中...')
            Feature.objects.all().delete()
            self.stdout.write(self.style.WARNING('既存の機能データを削除しました'))

        # 機能マスターデータ
        features_data = [
            {
                'name': 'AI計測集計',
                'feature_type': 'ai_measurement',
                'description': '写真から測定データを分析・集計する機能',
                'is_active': True
            },
            {
                'name': 'データ分析ダッシュボード',
                'feature_type': 'data_analysis',
                'description': '収集したデータを可視化・分析する機能',
                'is_active': True
            },
            {
                'name': 'レポート出力',
                'feature_type': 'reporting',
                'description': 'PDFやExcel形式でレポートを出力する機能',
                'is_active': True
            },
            {
                'name': 'ユーザー管理',
                'feature_type': 'user_management',
                'description': 'スタッフや利用者のアカウント管理機能',
                'is_active': True
            },
            {
                'name': '拠点管理',
                'feature_type': 'location_management',
                'description': '拠点情報や設定の管理機能',
                'is_active': True
            },
            {
                'name': 'システム設定',
                'feature_type': 'custom',
                'description': '各種システム設定を管理する機能',
                'is_active': True
            },
        ]

        created_count = 0
        updated_count = 0

        for feature_data in features_data:
            feature, created = Feature.objects.get_or_create(
                name=feature_data['name'],
                defaults=feature_data
            )
            
            if created:
                created_count += 1
                self.stdout.write(f'✅ 機能「{feature.name}」を作成しました')
            else:
                # 既存の場合は更新
                for key, value in feature_data.items():
                    setattr(feature, key, value)
                feature.save()
                updated_count += 1
                self.stdout.write(f'🔄 機能「{feature.name}」を更新しました')

        self.stdout.write(
            self.style.SUCCESS(
                f'\n機能マスターのシードデータ処理が完了しました！\n'
                f'作成: {created_count}件, 更新: {updated_count}件'
            )
        )