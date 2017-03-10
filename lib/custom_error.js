
module.exports = function CustomError(name, message) {
    this.name = name;
    this.message = message;
    // this.stack = console.trace(name);
    // function toString() {
        // return "Error: "+this.name;
    // }
}
require('util').inherits(module.exports, Error);

