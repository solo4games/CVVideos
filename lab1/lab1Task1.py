from graphviz import Digraph

# Создаем объект диаграммы
team_chart = Digraph(format='png')
team_chart.attr(rankdir='TB', size='32')

# Основная роль
team_chart.node('PM', 'Менеджер проекта')

# Подчиненные менеджера проекта
team_chart.node('BA', 'Аналитик требований')
team_chart.node('Arch', 'Архитектор')
team_chart.node('LeadDev', 'Ведущий разработчик')
team_chart.node('LeadQA', 'Ведущий тестировщик')
team_chart.node('DocSpec', 'Специалист по документации')

# Связь с менеджером
team_chart.edges([('PM', 'BA'), ('PM', 'Arch'), ('PM', 'LeadDev'), ('PM', 'LeadQA'), ('PM', 'DocSpec')])

# Подчиненные аналитика и архитектора
team_chart.node('ReqDoc', 'Документатор требований')
team_chart.edge('BA', 'ReqDoc')

team_chart.node('DBA', 'Проектировщик базы данных')
team_chart.node('TechSel', 'Эксперт по выбору технологий')
team_chart.edge('Arch', 'DBA')
team_chart.edge('Arch', 'TechSel')

# Разработчики под ведущим разработчиком
team_chart.node('Frontend', 'Frontend-разработчик')
team_chart.node('Backend', 'Backend-разработчик')
team_chart.node('Integrator', 'Инженер интеграции')
team_chart.edges([('LeadDev', 'Frontend'), ('LeadDev', 'Backend'), ('LeadDev', 'Integrator')])

# Тестировщики под ведущим тестировщиком
team_chart.node('AutoQA', 'Инженер автоматизации тестов')
team_chart.node('ManualQA', 'Ручной тестировщик')
team_chart.edges([('LeadQA', 'AutoQA'), ('LeadQA', 'ManualQA')])

# Сохраняем диаграмму
team_chart.render('team_hierarchy', view=True)