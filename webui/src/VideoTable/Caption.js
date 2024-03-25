import React, { Component } from 'react';

export var captionManager = {
	
	player: null,
	killbarrager: true,
	barragers: [],
	inter: 0,
	
	onload: function(player, video) {	

		this.player = player;
		this.index = video.index;
		this.parseCaptions(video.videoCaption);
		
		player.addEventListener('pause', () => {
			clearInterval(this.inter);
		});

		// player.addEventListener('play', () => {
		// 	if (this.player.paused) {
		// 		for(var i = 1;i < this.player.children.length;i++){
		// 			if(i>0){
		// 				this.player.removeChild(this.player.children[i]);
		// 				i--;
		// 			}
		// 		}
		// 		this.inter = setInterval(() => {
		// 			barragerMove(this.player, this.player.width);
		// 			barragerStart(this.player, this.player.currentTime, document.defaultView.getComputedStyle(this.player).height, this.barragers);
		// 			if (this.killbarrager) clearAll(player);
		// 		}, 250);
		// 	}
		// 	else {
		// 		clearInterval(this.inter);
		// 	}
		// });
		
		addHandler(player,"click", (event) => {
			if (this.player.paused) {
				for(var i = 1;i < this.player.parentElement.children.length;i++){
					if(i>0){
						this.player.parentElement.removeChild(this.player.parentElement.children[i]);
						i--;
					}
				}
				this.inter = setInterval(() => {
					barragerMove(this.player, document.defaultView.getComputedStyle(this.player).width);
					barragerStart(this.index, this.player, this.player.currentTime, document.defaultView.getComputedStyle(this.player).height, this.barragers);
					if (this.killbarrager) clearAll(player);
				}, 250);
			}
			else {
				clearInterval(this.inter);
			}
		});
	},

	toggleCaption: function (video) {
		this.killbarrager = true;
	},

	parseCaptions: function(caption){
		const lines = caption.split('\n');
  		// Process every 4th line
		this.barragers = lines.filter((_, index) => index % 4 === 1).map(line => {
    		const parts = line.split(':');
			return {
				time: parseFloat(parts[0]),
				text: parts.slice(2).join(':'), // Join back any ':' found after splitting (in case the caption itself contains ':'),
				color: "yellow",
				location: Math.floor(Math.random() * 101)
			};
  		});	
	}	
}

function barragerMove(player, width){
	var container = player.parentElement;	
	for(var i = 1;i < container.children.length;i++){
		if(i>0){
			var child = container.children[i];
			var childwidth = (document.defaultView.getComputedStyle(child)).width;
			
			if(parseInt(child.style.right) + parseInt(childwidth) > parseInt(width) - 30){
				container.removeChild(child);
				i--;
			}else{
				child.style.right = (parseInt(child.style.right) + 20) + "px";
			}
		}
	}
}

function barragerStart(index, player, curtime, height, barragers){
	const colors = ["red", "yellow", "blue", "green"];
	var hi = parseInt(height);
	var container = player.parentElement;
	var container = document.getElementsByClassName('video')[index];
	for(var i in barragers){
		if(barragers[i].time - curtime > -0.125 && barragers[i].time - curtime < 0.125){
			// emission
			if (barragers[i].text.trimStart().startsWith("Speaker")) continue;

			var node = document.createElement('div');
			var textnode = document.createTextNode(barragers[i].text);
			node.appendChild(textnode);
			node.style.position = 'absolute';
			node.style.top = Math.ceil(Math.random() * 0.6 * (hi - 20)) + "px";
			node.style.right = 0;
			node.style.color = colors[Math.floor(Math.random() * colors.length)];;
			container.appendChild(node);
			
		}
	}

}

function clearAll(player){
	var container = player.parentElement;
	for(var i = 1;i < container.children.length;i++){
		if(i>0){
			container.removeChild(container.children[i]);
			i--;
		}
	}
}

function addHandler(element, type, handler){
	if (element?.addEventListener != null){
		element.addEventListener(type, handler, false);
	} else if (element?.attachEvent){
		element.attachEvent("on" + type, handler);
	} else {
		if (element != null) element["on" + type] = handler;
	}
}






