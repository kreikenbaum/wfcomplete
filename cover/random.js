// pseudo-random string of length length (at least one)
exports.string = length => string(length);

// courtesy of stackoverflow.com/questions/1349404
function string(length) {
    if ( length <= 0 ) { length = 1; }
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for( var i=0; i < length; i++ )
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}
