if (!localStorage.on) {
    localStorage.on = '1';
}

if (localStorage.on == '1') {
	chrome.browserAction.setIcon({path: "images/icon_HB19.png"});
} else {
	chrome.browserAction.setIcon({path: "images/icon_HB19.png"});
}

chrome.browserAction.onClicked.addListener(function(tab) {
	if (localStorage.on == '1') {
		chrome.browserAction.setIcon({path: "images/icon_HB19.png"});
		localStorage.on = '0';
	} else {
		chrome.browserAction.setIcon({path: "images/icon_HB19.png"});
		localStorage.on = '1';
	}
});

chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
	if (localStorage.on == '1') {
		chrome.tabs.insertCSS(null, {code: "img{}", runAt: "document_start"});	
		chrome.storage.sync.set({'localStorageIsOn': "1"}, function() { });    
	} else 
		chrome.storage.sync.set({'localStorageIsOn': "0"}, function() { });   
});



