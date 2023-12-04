import argparse
import json
from   openai import OpenAI
import sqlite3
import sys
import tiktoken
import time

model = 'gpt-4'
client = OpenAI()

'''
texts = [
# allsources
"私はAIとして、人間のように感情や個人的な経験を持っていません。私は訓練されたデータに基づいてテキストを処理し、生成することができます。私の目的は、ユーザーからの入力に対応し、必要に応じて関連情報を提供することです。",
"人工知能として、私は個人的な好みや感情を持っていません。しかし、多くのユーザーが「進撃の巨人」や「鬼滅の刃」などのアニメを楽しんでいるようです。",
"""1. ハイキュー!!: このアニメは、熱血高校バレーボールの物語で、個性豊かな登場人物と壮大なスポーツドラマが特徴です。

2. 進撃の巨人: このアニメは、人類が壁に閉じ込められた世界で生き残る戦いを描いています。サスペンス、アクション、ファンタジー要素が見事に融合しています。

3. 約束のネバーランド: このアニメは、死後の世界を舞台にした心温まる物語です。主人公の少女が亡くなった兄弟を探す旅に出ます。

4. 銀魂: このアニメは、時代劇風のコメディとアクションが組み合わさった物語です。主人公は、地球を守るために日本神話の神々と戦う必要があります。

5. エヴァンゲリオン: このアニメは、宇宙的な規模で展開する複雑なプロットと深遠なテーマで知られています。""",
"""1873年に建設が開始した東京駅は、日本最初の国有鉄道（JR）の中心地として機能しました。当時は木造で、翌年に完成しました。この駅は、大正時代に改装され、豪華な外観と内部に変身しました。

第二次世界大戦後、東京駅は再び復元され、現在見られる形になりました。その後、1964年に新宿駅がオープンし、東京駅は主要な商業および交通ハブから一般的な終点駅へと移行しました。

2013年には、東京駅は「グランドニックス」と呼ばれる巨大複合施設の一部となり、高級ホテルやレストラン、映画館、美術館、屋上庭園などが含まれています。

今日、東京駅は日本で最も繁華街のひとつであり続けており、約50万人の乗客が利用しています。""",

# no-eval
"私はAIとして、人間のように感情や個人的な経験を持っていません。私はデータに基づいて学び、誰かが私に提供した情報に基づいて対することができます。私の目的は、ユーザーエクスペリエンスを向上させ、必要に応じて問題解決を支援することです。",
"AIとして、私には個人的な好みや感情がないため、特定のアニメを持ってこなせません。しかし、多くの人々が「進撃の巨人」や「鬼滅刃」などのアニメを楽しんでいるようです。",
"""1. 「デビルズライン」: このアニメは、人間と竜に取り込まれた少年が、自分の過去と戦っていく物語です。血と死のテーマが強調されおり、「鬼滅の刃」や「チェンソーマン」のような作品に近い感じです。

2. 「キングダム」: こちらは中国の歴史的時代を舞台にした戦争アニメです。主人公は、苦境に立たされた家族を守るために戦い続ける子です。「鬼滅の刃」のような家族愛と侍の精神を描いた物語です。

3. 「進撃の巨人」: このアニメは、巨大な壁に閉じ込められた世界を舞台にしており、その外には巨人が住んでいます。人類は壁内で生し、壁を守る兵士たちが存在します。「鬼滅の刃」のようなサバイバル要素と、人間とモンスターの戦いを描いた物語です。

4. 「ハイキュー!!」: 高校バレーボールを題材にしたこのアニメは、個性豊かな登場人物たちが集結し、一緒に成長していく様子を描いいます。「鬼滅の刃」のような団結力と友情を描いた物語です。

5. 「約束のネバーランド」: このアニメは、地球上のどこにも行けない不思議な島に住む少女たちの物語です。彼女たちは、島の秘密をき明かし、故郷への帰還を夢見ています。「鬼滅の刃」のような探求と友情を描いた物語です。""",
"1873年に建設が開始した東京駅は、日本最初の国有鉄道（JR）の中心地として機能しました。当時の建築家である岡田武市氏によってデインされたこの駅は、1964年に火災に見舞われ、その後再建されました。現在の構造は、1990年代半ばから2000年代初頭にかけて行わた改修プロジェクト「新東京駅」の一部です。このプロジェクトは、高速電車や特急列車の発着場所としてのみならず、商業施設やオィスビルも含む複合的な都市型交通ハブへと進化しました。",

# airoboros
"私は人工知能として、コンピュータサーバー上に存在し、ユーザーからのテキスト入力に基づいて応答するように設計されています。私学習や改善ができず、私の行動はプログラミングされたものだけで構成されています。",
"AIとして、私には個人的な好みや感情がなく、訓練されたデータセットに基づいて動作するように設計されています。したがって、私はニメを視聴したり楽しんだりする能力を持ち合わせていません。",
"""1. 「デビルズライン」: このアニメは、人間と竜に支配された世界で、主人公が自分の過去と戦っていく物語です。「鬼滅の刃」や「チンソーマン」のようなダークファンタジー要素があります。

2. 「進撃の巨人」: このアニメは、人類が壁の外に住む巨大な生き物に脅かされる世界で展開されます。「鬼滅の刃」のようなサバイバホラー要素があります。

3. 「ハイキュー!!」: このアニメは、高校バレーボール部の青春群像劇です。「鬼滅の刃」のようなテーマ別エピソード構成と同じよう、各話は個々の登場人物の視点から始まります。

4. 「約束のネバーランド」: このアニメは、死後の世界で少年と少女が出会う物語です。「チェンソーマン」のような不気味な雰囲気とロテスクな描写が特徴的です。

5. 「暗殺教室」: このアニメは、暗殺者としての才能を持つ高校生が、暗殺者としての才能を持つ教師から学んでいく物語です。「鬼滅刃」のような格闘技の要素があります。""",
"1873年に建設が開始した東京駅は、日本最初の国有鉄道（JR）の中心地として機能しました。当時は「大名」と呼ばれており、その後「橋」と改称されました。1914年に現在の場所に移転し、1960年代に再建されました。今日、東京駅は世界で最も繁華街のひとつであり商業施設やレストラン、ホテル、美術館などが入っている複合施設です。",
]
'''

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        print('---')
        self.print_help()
        sys.exit(2)

# parser = argparse.ArgumentParser(description="Default Reply Benchmarker")
parser = CustomArgumentParser()
parser.add_argument('-d', required=True, help='Require a data file')
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Error: {e}")
    parser.print_help()
    exit(1)

data_file = args.d
texts = json.load(open(data_file))


# Define OpenAI function
def call_openai(text):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Japanese translator. Please analyze whether the Japanese is used completely correctly and naturally for a native speaker. Please point out only inaccuracies in language usage: missing words, grammar, punctuation, improper tone/politeness, etc, and ignore the content of the submissions completely.",
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        temperature=0.1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(completion)
    return completion.choices[0].message.content

for text in texts:
    print()
    print("===")
    print()
    print(text)
    print('---')
    response = call_openai(text)
    print(response)