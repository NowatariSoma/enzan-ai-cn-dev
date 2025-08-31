from django.core.management.base import BaseCommand
from locations.models import Location, Feature, LocationFeature


class Command(BaseCommand):
    help = '拠点と機能の関連付けデータを作成します'

    def handle(self, *args, **options):
        """
        各拠点に適切な機能を関連付けます
        """
        created_count = 0
        updated_count = 0

        # =================================================================
        # 拠点ごとの機能設定定義（階層構造対応）
        # =================================================================
        # 各拠点の工事段階や設備に応じて機能の有効/無効を設定
        # - True: 機能が利用可能
        # - False: 機能が無効（フロントエンドでボタンがグレーアウト）
        # 
        # 機能タイプとフロントエンドマッピング:
        # ai_a_measurement → aiMeasurement (親機能)
        # measurement → measurement (A計測集計)
        # simulation → simulation (最終変位・沈下予測)
        # modelCreation → modelCreation (予測モデル作成)
        # reporting → reportGeneration (レポート出力)
        
        location_feature_mapping = {
            # 北海道阿寒: フル機能対応の試験拠点
            '01-hokkaido-akan': {
                'ai_a_measurement': True,  # AI-A計測システム全体
                'measurement': True,       # A計測集計
                'simulation': True,        # 最終変位・沈下予測
                'modelCreation': True,     # 予測モデル作成
                'reporting': True,         # レポート出力
            },
            
            # 北海道厚賀: フル機能対応の本格運用拠点
            '02-hokkaido-atsuga': {
                'ai_a_measurement': True,  # AI-A計測システム全体
                'measurement': True,       # A計測集計
                'simulation': True,        # 最終変位・沈下予測
                'modelCreation': True,     # 予測モデル作成
                'reporting': True,         # レポート出力
            },
            
            # 東北蔵王: 基本機能のみ（レポート・モデル作成は開発中）
            '03-tohoku-zao': {
                'ai_a_measurement': True,  # AI-A計測システム全体
                'measurement': True,       # A計測集計
                'simulation': True,        # 最終変位・沈下予測
                'modelCreation': False,    # モデル作成機能は開発中
                'reporting': False,        # レポート機能は開発中
            },
            
            # 関東箱根: 基本計測のみ（予測機能は計画段階）
            '04-kanto-hakone': {
                'ai_a_measurement': True,  # AI-A計測システム全体
                'measurement': True,       # A計測集計
                'simulation': False,       # 最終変位・沈下予測は計画段階
                'modelCreation': False,    # モデル作成機能は計画段階
                'reporting': True,         # レポート出力
            },
            
            # 中部富士: フル機能対応の主要拠点
            '05-chubu-fuji': {
                'ai_a_measurement': True,  # AI-A計測システム全体
                'measurement': True,       # A計測集計
                'simulation': True,        # 最終変位・沈下予測
                'modelCreation': True,     # 予測モデル作成
                'reporting': True,         # レポート出力
            }
        }

        # すべての拠点に機能を関連付け
        for location in Location.objects.all():
            self.stdout.write(f'\n拠点: {location.name} ({location.location_id})')
            
            # この拠点の機能設定を取得
            location_settings = location_feature_mapping.get(location.location_id, {})
            
            # 全ての機能をチェック
            for feature in Feature.objects.filter(is_active=True):
                feature_type = feature.feature_type
                is_enabled = location_settings.get(feature_type, False)
                
                # LocationFeatureを作成または更新
                location_feature, created = LocationFeature.objects.get_or_create(
                    location=location,
                    feature=feature,
                    defaults={
                        'is_enabled': is_enabled,
                        'settings': {}
                    }
                )
                
                if created:
                    created_count += 1
                    status = "✅ 作成" if is_enabled else "➖ 作成(無効)"
                    self.stdout.write(f'  {status}: {feature.name}')
                else:
                    # 既存の場合は有効/無効状態を更新
                    if location_feature.is_enabled != is_enabled:
                        location_feature.is_enabled = is_enabled
                        location_feature.save()
                        updated_count += 1
                        status = "🔄 更新→有効" if is_enabled else "🔄 更新→無効"
                        self.stdout.write(f'  {status}: {feature.name}')
                    else:
                        status = "✅ 有効" if is_enabled else "➖ 無効"
                        self.stdout.write(f'  {status}: {feature.name}')

        self.stdout.write(
            self.style.SUCCESS(
                f'\n拠点機能の関連付けが完了しました！\n'
                f'作成: {created_count}件, 更新: {updated_count}件'
            )
        )
