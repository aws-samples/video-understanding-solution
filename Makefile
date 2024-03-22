bootstrap:
	./utils/bootstrap.sh

prepare:
	OS=$$(uname -a | sed -n 's/^\([^ ]*\) .*/\1/p') ;\
	if [ "$$OS" = "Darwin" ]; then ./utils/install_prerequisites_for_macos_ventura.sh ;\
	elif [ "$$OS" = "Linux" ]; then @VER=$$(uname -a | sed -n 's/.*amzn\([0-9]*\).*/\1/p') ; if [ "$$VER" = "2" ]; then ./utils/install_prerequisites_for_amazon_linux_2.sh ; elif [ "$$VER" = "2023" ]; then ./utils/install_prerequisites_for_amazon_linux_2023.sh ; else echo "OS is not currently supported" ; fi;\
	else echo "OS is not currently supported" ;\
	fi
  
deploy:
	OS=$$(uname -a | sed -n 's/^\([^ ]*\) .*/\1/p') ;\
	if [ "$$OS" = "Darwin" ]; then ./utils/deploy.sh ;\
	elif [ "$$OS" = "Linux" ]; then sg docker -c './utils/deploy.sh' ;\
	else echo "OS not currently supported" ;\
	fi

destroy:
	./utils/destroy.sh

scan:
	./utils/scan.sh

bootstrap_and_deploy: bootstrap deploy