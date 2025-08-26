from django.core.management.base import BaseCommand
from locations.models import Feature, Location, LocationFeature
from decimal import Decimal
from datetime import date


class Command(BaseCommand):
    help = 'フロントエンドモックデータをベースにした拠点データを作成します'

    def add_arguments(self, parser):
        parser.add_argument(
            '--delete',
            action='store_true',
            help='既存のデータを削除してから作成します',
        )

    def handle(self, *args, **options):
        if options['delete']:
            self.stdout.write('既存の拠点データを削除中...')
            Location.objects.all().delete()  # LocationFeatureも一緒に削除される
            self.stdout.write(self.style.WARNING('既存の拠点データを削除しました'))

        # フロントエンドモックデータベースの拠点データ
        locations_data = [
            {
                'location_id': '01-hokkaido-akan',
                'name': '北海道阿寒',
                'region': '北海道',
                'prefecture': '北海道',
                'tunnel_name': '阿寒トンネル',
                'description': '国道240号線の山岳トンネル工事',
                'folder_name': '01-hokkaido-akan',
                'status': 'active',
                'start_date': date(2024, 4, 1),
                'total_length': 1157,
                'progress': Decimal('78.5'),
                'measurement_count': 64,
                'alert_level': 'normal',
                'latitude': Decimal('43.4500'),
                'longitude': Decimal('144.0167'),
                'features': ['ai_measurement', 'data_analysis', 'reporting', 'custom']
            },
            {
                'location_id': '02-hokkaido-atsuga',
                'name': '北海道厚賀',
                'region': '北海道',
                'prefecture': '北海道',
                'tunnel_name': '厚賀トンネル',
                'description': '日高自動車道延伸工事',
                'folder_name': '01-hokkaido-atsuga',
                'status': 'active',
                'start_date': date(2024, 6, 15),
                'total_length': 2450,
                'progress': Decimal('45.2'),
                'measurement_count': 48,
                'alert_level': 'warning',
                'latitude': Decimal('42.1083'),
                'longitude': Decimal('142.5639'),
                'features': ['ai_measurement', 'data_analysis', 'custom']
            },
            {
                'location_id': '03-tohoku-zao',
                'name': '東北蔵王',
                'region': '東北',
                'prefecture': '宮城県',
                'tunnel_name': '蔵王トンネル',
                'description': '東北自動車道バイパス工事',
                'folder_name': '01-hokkaido-akan',
                'status': 'monitoring',
                'start_date': date(2023, 9, 1),
                'total_length': 3200,
                'progress': Decimal('92.3'),
                'measurement_count': 72,
                'alert_level': 'normal',
                'latitude': Decimal('38.1000'),
                'longitude': Decimal('140.5667'),
                'features': ['ai_measurement', 'data_analysis', 'custom', 'reporting']
            },
            {
                'location_id': '04-kanto-hakone',
                'name': '関東箱根',
                'region': '関東',
                'prefecture': '神奈川県',
                'tunnel_name': '新箱根トンネル',
                'description': '国道1号線バイパストンネル',
                'folder_name': '04-kanto-hakone',
                'status': 'active',
                'start_date': date(2024, 2, 1),
                'total_length': 1850,
                'progress': Decimal('63.7'),
                'measurement_count': 56,
                'alert_level': 'normal',
                'latitude': Decimal('35.2333'),
                'longitude': Decimal('139.0167'),
                'features': ['ai_measurement', 'data_analysis', 'custom']
            },
            {
                'location_id': '05-chubu-fuji',
                'name': '中部富士',
                'region': '中部',
                'prefecture': '静岡県',
                'tunnel_name': '富士山麓トンネル',
                'description': '新東名高速道路延伸工事',
                'folder_name': '05-chubu-fuji',
                'status': 'active',
                'start_date': date(2024, 1, 15),
                'total_length': 4500,
                'progress': Decimal('34.8'),
                'measurement_count': 88,
                'alert_level': 'warning',
                'latitude': Decimal('35.3606'),
                'longitude': Decimal('138.7274'),
                'features': ['ai_measurement', 'data_analysis']
            },
        ]

        created_count = 0
        updated_count = 0

        for location_data in locations_data:
            features_config = location_data.pop('features')  # 機能設定を分離
            
            location, created = Location.objects.get_or_create(
                location_id=location_data['location_id'],
                defaults=location_data
            )
            
            if created:
                created_count += 1
                self.stdout.write(f'✅ 拠点「{location.name}」を作成しました')
            else:
                # 既存の場合は更新
                for key, value in location_data.items():
                    setattr(location, key, value)
                location.save()
                updated_count += 1
                self.stdout.write(f'🔄 拠点「{location.name}」を更新しました')

            # 機能設定を追加
            self._setup_location_features(location, features_config)

        self.stdout.write(
            self.style.SUCCESS(
                f'\n拠点のシードデータ処理が完了しました！\n'
                f'作成: {created_count}件, 更新: {updated_count}件'
            )
        )

    def _setup_location_features(self, location, features_config):
        """拠点の機能設定を行う"""
        # 既存の機能設定をクリア
        LocationFeature.objects.filter(location=location).delete()

        if features_config == 'all':
            # 全機能を追加
            features = Feature.objects.filter(is_active=True)
        else:
            # 指定された機能タイプの機能を追加
            features = Feature.objects.filter(
                feature_type__in=features_config,
                is_active=True
            )

        feature_count = 0
        for feature in features:
            LocationFeature.objects.create(
                location=location,
                feature=feature,
                is_enabled=True
            )
            feature_count += 1

        self.stdout.write(f'  └─ {feature_count}個の機能を設定しました')