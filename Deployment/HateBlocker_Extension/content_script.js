
function handleIMGS(textNode) {
	try {
		var _allImagesList_ = textNode.getElementsByTagName('img');

		for(var i = 0; i < _allImagesList_.length; i++)
		{
			var address = _allImagesList_[i].src;

			var jsonIssues = {};
			$.ajax({
				url: "http://127.0.0.1:5000/?url=".concat(address),
				async: false,
				dataType: 'json',
				success: function(data) {
					jsonIssues = data.body;
				}
			});

			x = parseFloat(jsonIssues.replace('"', ''))

			if (x > 0.5){
				console.log(x, address);

				_allImagesList_[i].classList.add("needtoloadimagestyle");
				// _allImagesList_[i].addEventListener('click', function (e) {
				// 	var img = document.createElement('img');
				// 	var timestamp = new Date().getTime();
				// 	this.src=this.src+"?i_fix="+timestamp;
				// 	this.classList.remove("needtoloadimagestyle");
				// });
			}

			_allImagesList_[i].addEventListener('click', function (e) {
				var img = document.createElement('img');
				var timestamp = new Date().getTime();
				this.src=this.src+"?i_fix="+timestamp;
				this.classList.remove("needtoloadimagestyle");
			});

	}

	} catch (error){
		console.log("e: "+error);
	}
}
var localStorageIsOn="0"
chrome.storage.sync.get('localStorageIsOn', function(items){
    localStorageIsOn = items['localStorageIsOn'];
	if (localStorageIsOn=="1") {
		handleIMGS(document);
	}
	chrome.storage.local.clear(function() {});
});