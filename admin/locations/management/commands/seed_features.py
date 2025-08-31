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

        # 機能マスターデータ（階層構造）
        # フロントエンド側の期待値:
        # - aiMeasurement (親): AI-A計測システム全体
        # - measurement (子): A計測集計
        # - simulation (子): 最終変位・沈下予測
        # - modelCreation (子): 予測モデル作成
        # - reportGeneration: レポート出力
        # - userManagement: ユーザー管理
        # - locationManagement: 拠点管理
        
        features_data = [
            # ==========================================
            # 親機能: AI-A計測システム
            # ==========================================
            {
                'name': 'AI-A計測集計',
                'feature_type': 'ai_a_measurement',  # → フロントエンド: aiMeasurement
                'description': 'AI技術を活用した包括的な計測・分析システム（親機能）',
                'is_active': True,
                'parent_feature': None,
                'display_order': 10
            },
            
            # ==========================================
            # AI-A計測の子機能群
            # ==========================================
            {
                'name': 'A計測集計',
                'feature_type': 'measurement',  # → フロントエンド: measurement
                'description': 'トンネル掘削時の変位・沈下データの収集と可視化機能',
                'is_active': True,
                'parent_feature': 'AI-A計測集計',
                'display_order': 11
            },
            {
                'name': '最終変位・沈下予測',
                'feature_type': 'simulation',  # → フロントエンド: simulation
                'description': '掘削完了時の最終変位・沈下量の予測シミュレーション機能',
                'is_active': True,
                'parent_feature': 'AI-A計測集計',
                'display_order': 12
            },
            {
                'name': '予測モデル作成',
                'feature_type': 'modelCreation',  # → フロントエンド: modelCreation
                'description': 'AI機械学習による変位・沈下予測モデルの作成・学習機能',
                'is_active': True,
                'parent_feature': 'AI-A計測集計',
                'display_order': 13
            },
            
            # ==========================================
            # その他の独立機能
            # ==========================================
            {
                'name': 'レポート出力',
                'feature_type': 'reporting',  # → フロントエンド: reportGeneration
                'description': '計測結果や分析データのPDF・Excel形式でのレポート出力機能',
                'is_active': True,
                'parent_feature': None,
                'display_order': 20
            },
            {
                'name': 'ユーザー管理',
                'feature_type': 'user_management',  # → フロントエンド: userManagement
                'description': 'システム利用者のアカウント管理・権限設定機能',
                'is_active': True,
                'parent_feature': None,
                'display_order': 30
            },
            {
                'name': '拠点管理',
                'feature_type': 'location_management',  # → フロントエンド: locationManagement
                'description': '工事拠点の情報管理・設定機能',
                'is_active': True,
                'parent_feature': None,
                'display_order': 40
            },
        ]

        created_count = 0
        updated_count = 0

        # 親機能を先に作成
        parent_features = {}
        for feature_data in features_data:
            if feature_data.get('parent_feature') is None:
                parent_feature, created = Feature.objects.get_or_create(
                    name=feature_data['name'],
                    defaults={
                        'feature_type': feature_data['feature_type'],
                        'description': feature_data['description'],
                        'is_active': feature_data['is_active'],
                        'display_order': feature_data.get('display_order', 0),
                        'parent_feature': None
                    }
                )
                parent_features[feature_data['name']] = parent_feature
                if created:
                    created_count += 1
                    self.stdout.write(f'✅ 機能「{feature_data["name"]}」を作成しました')
                else:
                    # 既存の場合は更新
                    parent_feature.feature_type = feature_data['feature_type']
                    parent_feature.description = feature_data['description']
                    parent_feature.is_active = feature_data['is_active']
                    parent_feature.display_order = feature_data.get('display_order', 0)
                    parent_feature.save()
                    updated_count += 1
                    self.stdout.write(f'🔄 機能「{feature_data["name"]}」を更新しました')

        # 子機能を作成
        for feature_data in features_data:
            if feature_data.get('parent_feature') is not None:
                parent_name = feature_data['parent_feature']
                parent_obj = parent_features.get(parent_name)
                
                if parent_obj:
                    child_feature, created = Feature.objects.get_or_create(
                        name=feature_data['name'],
                        defaults={
                            'feature_type': feature_data['feature_type'],
                            'description': feature_data['description'],
                            'is_active': feature_data['is_active'],
                            'display_order': feature_data.get('display_order', 0),
                            'parent_feature': parent_obj
                        }
                    )
                    if created:
                        created_count += 1
                        self.stdout.write(f'✅ 子機能「{feature_data["name"]}」を作成しました（親: {parent_name}）')
                    else:
                        # 既存の場合は更新
                        child_feature.feature_type = feature_data['feature_type']
                        child_feature.description = feature_data['description']
                        child_feature.is_active = feature_data['is_active']
                        child_feature.display_order = feature_data.get('display_order', 0)
                        child_feature.parent_feature = parent_obj
                        child_feature.save()
                        updated_count += 1
                        self.stdout.write(f'🔄 子機能「{feature_data["name"]}」を更新しました（親: {parent_name}）')
                else:
                    self.stdout.write(self.style.ERROR(f'❌ 親機能「{parent_name}」が見つかりません'))

        self.stdout.write(
            self.style.SUCCESS(
                f'\n機能マスターのシードデータ処理が完了しました！\n'
                f'作成: {created_count}件, 更新: {updated_count}件'
            )
        )