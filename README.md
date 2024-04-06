# ML для анализа почвы со спутника

В текущий момент анализ почвы во многих областях РФ производится по старинке: выкапывается земля, берутся образцы, делается анализ на содержание тех или иных веществ. 
Такой подход проверен временем и работает, позволяет делать выводы о пригодности почвы к засеиванию. Однако у такого подхода есть и существенный недостаткок - этот процесс
очень дорого повторять раз за разом на различных территориях, а также это требует больших временных затрат.

Спутники регулярно делают снимки планеты и могут предоставить собранную информацию. Так вот целью моей работы является попытка использовать информацию со спутников для того, чтобы
заменить ручной труд, обучив модель предсказывать содержание в почве полезных веществ на основании цветовых данных, считанных датчиками спутника.

Моя работа не отягащена вынесением моделью вердикта о пригодности почвы, а лишь обучается по тестовым данным предсказывать содержание нескольких групп полезных веществ и строить графики.
Так в чем же в итоге польза от программы? Если модель окажется способной "неплохо" видеть закономерности в данных со спутника и отображать их, то это и будет первым шагом в сторону
автоматизации проверки почв.

К сожалению, тестовых данных было крайне мало, но по анализу некоторых веществ из почвы получилось получить подобные результаты:  

![image](https://github.com/vitbogit/university-dirt-ml/assets/61887732/e0c62a4b-a3b4-4550-a764-01133b53f277)

Что мы видим:
1) модель "увидела" явные высокие показатели в левой части, и уход ниже в правой части;
2) модель "запуталась" в правой части.

Что касается первого пункта - мы частично достигли своей цели, благодаря модели экологи смогли бы автоматически довольно точно "предсказать" некоторые показатели почвы в определенных условиях.
Что касается второго пункта - необходимо учесть, что на скриншоте представлен показатель лишь одного из почти десятка!!! веществ в почве, имеющих высокое значение для ее качества, а также по этому веществу 
на скриншоте представлен анализ лишь по одному из более чем десятка!!! датчиков спутника. Модель строилась так, чтобы учиться на всех веществах и по всем датчикам, но графики строятся по каждому веществу и датчику отдельно. Почему выбран такой формат вывода? Потому что, к сожалению, в тестовых данных не дается пометки о статусе почвы - пригодная / не пригодная, и решилась по сути не задача классификации, а задача регрессии.

Выводы:
- с подобным подходом модель способна генерировать некоторые возможно полезные графики для ряда веществ из почвы при определенных условиях наблюдения со спутника;
- отсутствие возможности обучать модель классифицировать почву как носящую тот или иной статус из-за недостаточных тестовых данных принудила к попытке "предсказать все сразу, и ничего одновременно",
  из-за чего сложно судить о реальной точности модели.
