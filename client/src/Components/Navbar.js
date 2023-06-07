import React from 'react'
import Header1 from './Header1'
import Button from './Button'


function Navbar({children, theme, appname, apphome}) {
  return (
    <div className={'navbar navbar-expand-lg navbar-'+ theme + 'bg-' + theme}>
        <a className='navbar-brand' href={apphome}>{appname}</a>
            <Button className="navbar-toggler"></Button>
    </div>
  )
}

export default Navbar