import smtplib
from getpass import getpass
import datetime

def encaminhar_email(origem, senha, destino, assunto, mensagem):
    try:
        # Configurações do servidor SMTP
        servidor_smtp = 'smtp.gmail.com'
        porta_smtp = 587

        # Criação da conexão com o servidor SMTP
        conexao = smtplib.SMTP(servidor_smtp, porta_smtp)
        conexao.starttls()

        # Autenticação com as credenciais
        conexao.login(origem, senha)

        # Envio do e-mail
        cabecalho = f"From: {origem}\nTo: {destino}\nSubject: {assunto}\n"
        mensagem_completa = cabecalho + mensagem
        conexao.sendmail(origem, destino, mensagem_completa.encode('utf-8'))

        # Encerramento da conexão com o servidor SMTP
        conexao.quit()

        print("E-mail enviado com sucesso!")
    except smtplib.SMTPAuthenticationError:
        print("Erro de autenticação: Nome de usuário ou senha inválidos.")
    except smtplib.SMTPException as e:
        print("Erro ao enviar o e-mail:", e)


# Função para gerar o corpo do e-mail com base no dia da semana
def gerar_corpo_email():
    dia_semana = datetime.datetime.today().weekday()

    if dia_semana == 0:  # Segunda-feira
        mensagem = "Bom início de semana!"
    elif dia_semana == 4:  # Sexta-feira
        mensagem = "Bom fim de semana!"
    else:
        mensagem = "Dentro de .github/workflows!"

    return mensagem

# Função para enviar o e-mail
def enviar_email():
    origem = 'seriestemporaiss@gmail.com'
    senha = 'aloqkbbsftrgjiol'
    destino = ['t217517@dac.unicamp.br','m236226@dac.unicamp.br'] # m236226@dac.unicamp.br, tiagohemont@hotmail.com
    assunto = 'Testando o envio de e-mail pelo Python'
    mensagem = gerar_corpo_email()

    # Encaminhar e-mail
    encaminhar_email(origem, senha, destino, assunto, mensagem)

enviar_email()
